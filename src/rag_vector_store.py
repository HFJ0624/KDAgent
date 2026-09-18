"""ChromaDB-based vector store for RAG."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .utils import ensure_dir, read_yaml


# Minimum fields that metadata must preserve
_METADATA_INT_KEYS = {"chunk_index"}


class RAGVectorStore:
    """A lightweight wrapper around a ChromaDB collection."""

    def __init__(self, rag_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        vs_cfg = rag_config.get("vector_store", {}) or {}
        self.store_type: str = vs_cfg.get("type", "chroma")
        self.persist_directory: str = vs_cfg.get("persist_directory", "outputs/chroma_db")
        self.collection_name: str = vs_cfg.get("collection_name", "swat_process_kb")

        self.embedding_cfg = rag_config.get("embedding", {}) or {}
        self.chunk_cfg = rag_config.get("chunking", {}) or {}
        self.retrieval_cfg = rag_config.get("retrieval", {}) or {}

        self.logger = logger or logging.getLogger("rag_vector_store")
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _ensure_dir(self) -> str:
        return ensure_dir(self.persist_directory)

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "无法导入 'chromadb'，请先执行 'pip install chromadb'。"
                ) from exc
            self._ensure_dir()
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def _get_collection(self, create: bool = False):
        if self._collection is not None:
            return self._collection
        client = self._get_client()
        if create:
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._collection = client.get_collection(name=self.collection_name)
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear_collection(self) -> None:
        """Delete the existing collection, used to rebuild from scratch."""
        if self._collection is not None:
            try:
                self._collection = None
            except Exception:
                pass
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
            self.logger.info("[rag_vector_store] 已删除集合：%s", self.collection_name)
        except Exception:
            self.logger.info("[rag_vector_store] 集合 %s 尚未创建，跳过。", self.collection_name)

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 500,
    ) -> None:
        """Insert or batch-update chunks.

        ChromaDB has an internal batch-size limit, so we chunk the batches here.
        """
        if not ids:
            return
        collection = self._get_collection(create=True)

        # Normalize metadata to one dict per document
        safe_metadatas: List[Dict[str, Any]] = []
        if metadatas:
            for m in metadatas:
                if not isinstance(m, dict):
                    safe_metadatas.append({})
                else:
                    # ChromaDB metadata supports only str/int/float/bool; convert as needed
                    safe_metadatas.append(self._sanitize_metadata(m))
        else:
            safe_metadatas = [{} for _ in ids]

        for i in range(0, len(ids), batch_size):
            chunk_ids = ids[i : i + batch_size]
            chunk_docs = documents[i : i + batch_size]
            chunk_embs = embeddings[i : i + batch_size]
            chunk_metas = safe_metadatas[i : i + batch_size]
            collection.upsert(
                ids=chunk_ids,
                documents=chunk_docs,
                embeddings=chunk_embs,
                metadatas=chunk_metas,
            )
        self.logger.info(
            "[rag_vector_store] 向集合 '%s' upsert 了 %d 个 chunk。",
            self.collection_name,
            len(ids),
        )

    def count(self) -> int:
        try:
            collection = self._get_collection(create=False)
            return collection.count()
        except Exception:
            return 0

    def query(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve the top_k documents closest to the query vector."""
        k = int(top_k or self.retrieval_cfg.get("top_k", 5))
        collection = self._get_collection(create=False)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        res = collection.query(**kwargs)

        # Normalize the output into a convenient dict
        ids = res.get("ids", [[]])
        docs = res.get("documents", [[]])
        metas = res.get("metadatas", [[]])
        dists = res.get("distances", [[]])

        ids_list = ids[0] if ids else []
        docs_list = docs[0] if docs else []
        metas_list = metas[0] if metas else []
        dists_list = dists[0] if dists else []

        retrieved: List[Dict[str, Any]] = []
        for i in range(len(ids_list)):
            dist_val = None
            if i < len(dists_list) and dists_list[i] is not None:
                try:
                    dist_val = float(dists_list[i])
                except Exception:
                    dist_val = None
            retrieved.append(
                {
                    "id": ids_list[i],
                    "content": docs_list[i] if i < len(docs_list) else "",
                    "metadata": metas_list[i] if i < len(metas_list) else {},
                    "distance": dist_val,
                    "score": self._distance_to_score(dist_val),
                }
            )
        return {"results": retrieved}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _distance_to_score(d: Optional[float]) -> Optional[float]:
        """ChromaDB returns cosine distance (0 = identical, 2 = completely opposite).

        Convert it to a similarity score in [0, 1]; returns None when d is None.
        """
        if d is None:
            return None
        # cosine similarity = 1 - distance
        sim = 1.0 - d
        if sim < 0.0:
            sim = 0.0
        if sim > 1.0:
            sim = 1.0
        return round(float(sim), 4)

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, int):
                out[k] = v
            elif isinstance(v, float):
                out[k] = v
            elif isinstance(v, (str,)):
                out[k] = v
            else:
                out[k] = str(v)
        return out


def load_vector_store_from_config(
    rag_config_path: str, logger: Optional[logging.Logger] = None
) -> RAGVectorStore:
    cfg = read_yaml(rag_config_path)
    return RAGVectorStore(cfg, logger=logger)
