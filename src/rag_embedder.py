"""RAG Embedding client.

Wraps an OpenAI-compatible Embedding API (e.g., SiliconFlow), providing:
  - Batch embedding
  - Automatic retry with exponential backoff
  - Primary/fallback model switching
  - API Key read from an environment variable (never hardcoded)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

from .utils import read_yaml


class RAGEmbedder:
    """OpenAI-compatible Embedding client used by the RAG pipeline."""

    def __init__(self, rag_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        emb_cfg = rag_config.get("embedding", {}) or {}
        self.provider: str = emb_cfg.get("provider", "openai_compatible")
        self.base_url: str = emb_cfg.get("base_url", "")
        self.model: str = emb_cfg.get("model", "text-embedding-3-small")
        self.fallback_model: Optional[str] = emb_cfg.get("fallback_model")
        self.batch_size: int = int(emb_cfg.get("batch_size", 8))
        self.timeout: int = int(emb_cfg.get("request_timeout_s", 60))
        self.max_retries: int = int(emb_cfg.get("max_retries", 5))

        env_name: str = emb_cfg.get("api_key_env", "SILICONFLOW_API_KEY")
        self.api_key: Optional[str] = os.environ.get(env_name)
        self.api_key_env = env_name

        self.logger = logger or logging.getLogger("rag_embedder")

        # Delay the import so this module can still be imported when openai is not
        # installed (e.g., when running smoke tests)
        self._client = None

    # ------------------------------------------------------------------
    # Public helper methods
    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        """Return whether both the API Key and base_url are ready."""
        if not self.base_url:
            return False
        if not self.api_key:
            return False
        return True

    # ------------------------------------------------------------------
    # Internal client management
    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "无法导入 'openai'，请先执行 'pip install openai'。"
                ) from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # Core Embedding API
    # ------------------------------------------------------------------
    def embed_texts(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Embed a list of texts using the given model.

        Returns a vector list aligned with the input; empty strings are skipped
        and mapped to an empty vector.
        """
        if not texts:
            return []
        chosen_model = model or self.model
        client = self._get_client()

        embeddings: List[List[float]] = []
        # Pre-fill placeholders for the string inputs
        placeholder: List[Optional[List[float]]] = [None] * len(texts)
        valid_indices: List[int] = []
        valid_texts: List[str] = []
        for i, t in enumerate(texts):
            if t is None or not str(t).strip():
                placeholder[i] = []
            else:
                valid_indices.append(i)
                valid_texts.append(str(t))

        if not valid_texts:
            return [e if e is not None else [] for e in placeholder]  # type: ignore[list-item]

        # Batch embedding + retry
        for batch_texts in self._chunkify(valid_texts, self.batch_size):
            batch_emb = self._embed_batch_with_retry(client, batch_texts, chosen_model)
            embeddings.extend(batch_emb)

        # Rebuild the aligned output
        result: List[List[float]] = []
        valid_cursor = 0
        for p in placeholder:
            if p is not None:
                result.append(p)  # type: ignore[arg-type]
            else:
                result.append(embeddings[valid_cursor])
                valid_cursor += 1
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        if text is None or not str(text).strip():
            return []
        return self.embed_texts([str(text)])[0]

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------
    def _embed_batch_with_retry(
        self, client, batch_texts: List[str], model: str
    ) -> List[List[float]]:
        """Embed a single batch with retry and primary/fallback model switching."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch_texts)
                data = getattr(resp, "data", None) or getattr(resp, "embeddings", None) or []
                vecs: List[List[float]] = []
                # Some providers return a list, others return individual objects
                for item in data:
                    if isinstance(item, dict):
                        vec = item.get("embedding") or item.get("vector")
                    else:
                        vec = getattr(item, "embedding", None) or getattr(item, "vector", None)
                    if vec is None:
                        raise RuntimeError("服务端返回结果中没有 embedding 字段。")
                    vecs.append(list(vec))
                if len(vecs) != len(batch_texts):
                    raise RuntimeError(
                        f"Embedding 数量不匹配：返回 {len(vecs)} vs 输入 {len(batch_texts)}"
                    )
                return vecs
            except Exception as exc:  # pragma: no cover - network errors take many forms
                last_exc = exc
                sleep_s = min(2 ** (attempt - 1), 16) + 0.1 * attempt
                self.logger.warning(
                    "[rag_embedder] 第 %d/%d 次尝试失败（model=%s）：%s。%.2fs 后重试...",
                    attempt, self.max_retries, model, exc, sleep_s,
                )
                time.sleep(sleep_s)

        # Switch to the fallback model
        if self.fallback_model and self.fallback_model != model:
            self.logger.warning(
                "[rag_embedder] 主模型失败，切换到备用模型 %s。",
                self.fallback_model,
            )
            try:
                return self._embed_batch_with_retry(
                    client, batch_texts, self.fallback_model
                )
            except Exception as fallback_exc:  # pragma: no cover
                last_exc = fallback_exc

        raise RuntimeError(f"Embedding 在 {self.max_retries} 次尝试后仍失败：{last_exc}")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def _chunkify(items: List[Any], size: int) -> Iterable[List[Any]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]


def load_embedder_from_config(
    rag_config_path: str, logger: Optional[logging.Logger] = None
) -> RAGEmbedder:
    """Create a RAGEmbedder from a YAML config."""
    cfg = read_yaml(rag_config_path)
    return RAGEmbedder(cfg, logger=logger)
