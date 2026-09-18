"""Build the SWaT RAG knowledge base.

Inputs
------
1. `variables_meta.csv`            -> variable-level knowledge chunks
2. `process_knowledge_template.md` -> process-level knowledge chunks
3. `data/rag_kb/*.md`              -> curated supplementary knowledge

Outputs
-------
Embeds the chunks and writes them into the ChromaDB collection configured in rag.yaml.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .rag_embedder import RAGEmbedder
from .rag_vector_store import RAGVectorStore
from .utils import ensure_dir, read_yaml


# ---------------------------------------------------------------
# Variable-name prefix dictionary (must match the README definition)
# ---------------------------------------------------------------
_VARIABLE_PREFIX_MEANING = {
    "FIT": "流量传感器",
    "LIT": "液位传感器",
    "AIT": "分析传感器",
    "PIT": "压力传感器",
    "DPIT": "差压传感器",
    "MV": "电动阀",
    "P": "泵",
    "UV": "紫外单元",
}


def _prefix_meaning(var_name: str) -> str:
    """Return the description of a variable prefix."""
    if not var_name:
        return "变量"
    upper = var_name.upper()
    # Order matters: longer prefixes must be matched first
    for p in ("DPIT", "AIT", "FIT", "LIT", "PIT", "MV", "UV", "P"):
        if upper.startswith(p):
            return f"{_VARIABLE_PREFIX_MEANING.get(p, '变量')} ({p})"
    return "变量"


# ---------------------------------------------------------------
# Chunk data structure
# ---------------------------------------------------------------
def _make_chunk(
    chunk_id: str,
    content: str,
    metadata: Dict[str, Any],
    source: str,
    doc_type: str,
) -> Dict[str, Any]:
    metadata = dict(metadata)
    metadata["source"] = source
    metadata["doc_type"] = doc_type
    return {
        "id": chunk_id,
        "content": content,
        "metadata": metadata,
    }


# ---------------------------------------------------------------
# Source 1: variables_meta.csv
# ---------------------------------------------------------------
def build_variable_chunks(variables_meta_csv: str) -> List[Dict[str, Any]]:
    """Generate knowledge chunks row by row from the variable metadata CSV.

    Each chunk describes the variable's name, type, stage, unit/semantics, and
    role in the process. It contains no case-specific information.
    """
    if not os.path.exists(variables_meta_csv):
        return []

    import pandas as pd  # local import so basic smoke tests can run without pandas

    df = pd.read_csv(variables_meta_csv)
    chunks: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        var_id = str(row.get("var_id", "") or "").strip()
        var_index = row.get("var_index", "")
        original_name = str(row.get("original_name", "") or "").strip()
        description = str(row.get("description", "") or "").strip()
        var_type = str(row.get("var_type", "") or "").strip()
        stage = str(row.get("stage", "") or "").strip()
        unit_or_semantics = str(row.get("unit_or_semantics", "") or "").strip()
        meaning = _prefix_meaning(original_name or "")

        text_parts = [
            f"变量 {original_name} (id={var_id}, index={var_index})。",
            f"含义：{meaning}。",
        ]
        if var_type:
            text_parts.append(f"类型：{var_type}。")
        if stage:
            text_parts.append(f"工段：{stage}。")
        if unit_or_semantics:
            text_parts.append(f"单位/语义：{unit_or_semantics}。")
        if description:
            text_parts.append(f"描述：{description}。")

        content = " ".join(text_parts)
        chunk = _make_chunk(
            chunk_id=f"var__{var_id}",
            content=content,
            metadata={
                "variable_name": original_name,
                "var_id": var_id,
                "var_index": str(var_index) if var_index != "" else "",
                "stage": stage,
                "variable_type": var_type,
                "unit_or_semantics": unit_or_semantics,
            },
            source="variables_meta.csv",
            doc_type="variable",
        )
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------
# Sources 2 & 3: markdown files
# ---------------------------------------------------------------
def _extract_source_front_matter(text: str) -> Tuple[str, Dict[str, str]]:
    """Extract the source metadata at the top of a Markdown file and remove it from the body.

    External material must be traceable, but Chroma metadata supports only scalar
    values. We use a concise YAML-style front matter so each external-knowledge
    file can declare its title, URL, DOI, access date, and source level; these
    fields are then copied into every knowledge chunk produced from that file.
    """
    if not text.startswith("---\n"):
        return text, {}

    end = text.find("\n---\n", len("---\n"))
    if end < 0:
        return text, {}

    metadata: Dict[str, str] = {}
    for line in text[len("---\n") : end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Keep only source metadata, to prevent arbitrary front matter from being written into the vector store.
        if key.startswith("source_") and value:
            metadata[key] = value

    return text[end + len("\n---\n") :].lstrip(), metadata


def chunk_markdown(
    text: str,
    source: str,
    doc_type: str,
    chunk_size: int,
    chunk_overlap: int,
    base_id_prefix: str,
) -> List[Dict[str, Any]]:
    """Split a markdown document into overlapping chunks."""
    if not text:
        return []

    text, source_metadata = _extract_source_front_matter(text)

    # Make retrieved fragments themselves carry a short citation, so the model can
    # judge the knowledge source from context; the metadata is used for raw JSONL
    # auditing and traceability checks in the paper.
    citation_prefix = ""
    source_title = source_metadata.get("source_title", "")
    source_url = source_metadata.get("source_url", "")
    if source_title:
        citation_prefix = f"[可核验来源] {source_title}"
        if source_url:
            citation_prefix += f" | {source_url}"
        citation_prefix += "\n"

    # Split by headings first to preserve structure
    heading_chunks = _split_by_headings(text)

    final_chunks: List[Dict[str, Any]] = []
    idx = 0
    for hunk in heading_chunks:
        # Split overly large paragraphs with overlap
        for sub in _split_with_overlap(hunk, chunk_size, chunk_overlap):
            idx += 1
            chunk_id = f"{base_id_prefix}__{idx:05d}"
            # If the text mentions a stage, record the stage label
            stage = _detect_stage(sub)
            final_chunks.append(
                _make_chunk(
                    chunk_id=chunk_id,
                    content=f"{citation_prefix}{sub}",
                    metadata={"stage": stage, **source_metadata},
                    source=source,
                    doc_type=doc_type,
                )
            )
    return final_chunks


def _split_by_headings(text: str) -> List[str]:
    """Split the text into multiple sections by markdown headings."""
    lines = text.splitlines()
    sections: List[str] = []
    buffer: List[str] = []
    for line in lines:
        if re.match(r"^#{1,6}\s", line):
            if buffer:
                sections.append("\n".join(buffer).strip())
                buffer = []
        buffer.append(line)
    if buffer:
        sections.append("\n".join(buffer).strip())
    return [s for s in sections if s]


def _split_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text with a character-level sliding window, avoiding an external tokenizer."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - chunk_overlap)
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap if chunk_overlap > 0 else end
        if start >= len(text):
            break
    return out


_STAGE_PATTERN = re.compile(r"\bP([1-6])\b")


def _detect_stage(text: str) -> str:
    m = _STAGE_PATTERN.search(text)
    if m:
        return f"P{m.group(1)}"
    return ""


# ---------------------------------------------------------------
# Orchestration entry
# ---------------------------------------------------------------
def collect_kb_chunks(
    data_dir: str,
    kb_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Aggregate all knowledge chunks from multiple sources."""
    log = logger or logging.getLogger("rag_kb_builder")
    chunks: List[Dict[str, Any]] = []

    # Source 1: variables_meta.csv
    var_csv = os.path.join(data_dir, "variables_meta.csv")
    if os.path.exists(var_csv):
        var_chunks = build_variable_chunks(var_csv)
        chunks.extend(var_chunks)
        log.info("[rag_kb_builder] 加入了 %d 个变量 chunk。", len(var_chunks))
    else:
        log.warning("[rag_kb_builder] variables_meta.csv 未找到：%s", var_csv)

    # Source 2: process_knowledge_template.md
    tpl_path = os.path.join(data_dir, "process_knowledge_template.md")
    if os.path.exists(tpl_path):
        with open(tpl_path, "r", encoding="utf-8") as f:
            text = f.read()
        tpl_chunks = chunk_markdown(
            text,
            source="process_knowledge_template.md",
            doc_type="process",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_id_prefix="process",
        )
        chunks.extend(tpl_chunks)
        log.info("[rag_kb_builder] 加入了 %d 个流程模板 chunk。", len(tpl_chunks))
    else:
        log.warning("[rag_kb_builder] process_knowledge_template.md 未找到：%s", tpl_path)

    # Source 3: data/rag_kb/*.md
    if os.path.isdir(kb_dir):
        md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
        for md_file in md_files:
            fname = os.path.basename(md_file)
            doc_type = _infer_doc_type(fname)
            with open(md_file, "r", encoding="utf-8") as f:
                text = f.read()
            md_chunks = chunk_markdown(
                text,
                source=fname,
                doc_type=doc_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                base_id_prefix=f"kb_{fname.replace('.md', '')}",
            )
            chunks.extend(md_chunks)
            log.info(
                "[rag_kb_builder] 从 %s 加入了 %d 个 chunk（类型：%s）。",
                fname,
                len(md_chunks),
                doc_type,
            )
    else:
        log.warning("[rag_kb_builder] kb_dir 未找到：%s", kb_dir)

    return chunks


def _infer_doc_type(filename: str) -> str:
    fname = filename.lower()
    if "process" in fname:
        return "process"
    if "relation" in fname:
        return "relation"
    if "pattern" in fname or "fault" in fname:
        return "fault_pattern"
    if "variable" in fname:
        return "variable"
    return "misc"


def build_kb(
    rag_config_path: str,
    data_dir: str,
    kb_dir: str,
    rebuild: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Main entry: build the entire RAG knowledge base and write it into ChromaDB."""
    log = logger or logging.getLogger("rag_kb_builder")
    cfg = read_yaml(rag_config_path)

    chunking = cfg.get("chunking", {}) or {}
    chunk_size = int(chunking.get("chunk_size", 800))
    chunk_overlap = int(chunking.get("chunk_overlap", 120))

    # 1. Collect the raw chunks --------------------------------------------
    chunks = collect_kb_chunks(
        data_dir=data_dir,
        kb_dir=kb_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        logger=log,
    )
    if not chunks:
        log.error("[rag_kb_builder] 未收集到任何 chunk，终止。")
        return {"num_chunks": 0}

    # 2. Embed the chunks ------------------------------------------------
    embedder = RAGEmbedder(cfg, logger=log)
    if not embedder.is_configured():
        raise RuntimeError(
            f"Embedding API 未配置。请设置环境变量 '{embedder.api_key_env}'，"
            f"并检查 {rag_config_path} 中的 'base_url'。"
        )

    documents: List[str] = [c["content"] for c in chunks]
    log.info("[rag_kb_builder] 正在对 %d 个 chunk 做 Embedding（batch_size=%d）...", len(documents), embedder.batch_size)
    embeddings = embedder.embed_texts(documents)

    # 3. Write into ChromaDB ------------------------------------------------
    store = RAGVectorStore(cfg, logger=log)
    if rebuild:
        store.clear_collection()
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    store.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    result = {
        "num_chunks": len(chunks),
        "collection_name": store.collection_name,
        "persist_directory": store.persist_directory,
    }
    log.info("[rag_kb_builder] 完成：%s", result)
    return result
