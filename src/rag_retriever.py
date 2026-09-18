"""RAG retriever for SWaT RCA.

Builds a query from each case's candidate variables and retrieves the top_k most
relevant knowledge chunks from the ChromaDB-based vector store.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from .rag_embedder import RAGEmbedder
from .rag_vector_store import RAGVectorStore
from .utils import read_yaml


# Industrial-domain keyword bank for focusing the retrieval direction (grouped by variable prefix)
# When a matching prefix is found in a variable name, description, or type, the related
# industrial terms are injected into the query to improve retrieval precision
_KEYWORD_BANK = {
    "flow": ["流量", "FIT", "入流", "出流", "flow sensor", "流量传感器"],
    "level": ["液位", "LIT", "储罐", "水位", "level sensor", "tank level"],
    "valve": ["阀门", "MV", "电动阀", "motorized valve", "valve state"],
    "pump": ["泵", "pump", "泵状态", "pump state", "水泵"],
    "pressure": ["压力", "PIT", "DPIT", "差压", "pressure sensor", "differential pressure"],
    "analyzer": ["分析仪", "AIT", "pH", "电导率", "浊度", "analyzer", "water quality"],
    "membrane": ["膜", "超滤", "反渗透", "RO", "UF", "membrane", "membrane fouling"],
    "uv": ["UV", "紫外", "消毒", "uv disinfection"],
}

# SWaT process-stage mapping: infer the process stage from the numeric suffix of a variable name
_STAGE_MAP = {
    "1": ("P1 取水", "P1 Intake", "raw water intake", "原水取水"),
    "2": ("P2 加药", "P2 Dosing", "chemical dosing", "加药系统"),
    "3": ("P3 超滤", "P3 UF", "ultrafiltration", "超滤膜"),
    "4": ("P4 UV", "P4 UV", "UV disinfection", "紫外消毒"),
    "5": ("P5 反渗透", "P5 RO", "reverse osmosis", "反渗透"),
    "6": ("P6 供水", "P6 Delivery", "water delivery", "供水系统"),
}


def build_rag_query(case: Dict[str, Any]) -> str:
    """Build a highly focused RAG query string from the case dictionary.

    Compared with a simple "SWaT root cause analysis for case X", this function:
    1. Extracts var_id, name, type, and description for each variable in top_k_variables;
    2. Finds the top variable with the highest case_level_score;
    3. Infers industrial keywords such as pump/valve/flow from the variable-name
       prefix (e.g., P302, MV301);
    4. Combines all of the above into a query containing concrete variable
       details and industrial terms, so the Embedding vector leans toward the
       current case's candidate relationships, retrieving more relevant
       knowledge chunks during vector search.
    """
    parts: List[str] = []

    case_id = case.get("case_id")
    task_desc = case.get("task", "SWaT 根因分析")
    parts.append(f"案例 {case_id} ({task_desc})")

    # --- 1. Extract detailed info from top_k_variables (new format) ---
    top_k_vars = case.get("top_k_variables") or []
    var_details: List[str] = []          # e.g. "V24/P302 (state)"
    var_names: List[str] = []            # the name field (real variable name)
    var_ids: List[str] = []              # the var_id field (e.g., V24)
    var_types: List[str] = []            # state / continuous etc.
    descriptions: List[str] = []         # variable semantic descriptions
    highest_score_var: str = ""
    highest_score: float = -1.0
    # Cumulative scores per stage, used to pick the process stage the case most cares about
    stage_scores: Dict[str, float] = {}

    if top_k_vars and isinstance(top_k_vars, list):
        for item in top_k_vars[:10]:
            if not isinstance(item, dict):
                continue

            var_id = str(item.get("var_id", "") or "")
            name = str(item.get("name", "") or "")
            vtype = str(item.get("type", "") or "")
            desc = str(item.get("description", "") or "")
            score = item.get("case_level_score")

            if name:
                var_names.append(name)
            if var_id:
                var_ids.append(var_id)
            if vtype:
                var_types.append(vtype)
            if desc:
                descriptions.append(desc)

            # Build the detailed variable description string, e.g. "V24/P302 (state)"
            detail_parts = []
            if var_id:
                detail_parts.append(var_id)
            if name and name != var_id:
                detail_parts.append(name)
            detail = "/".join(detail_parts) if detail_parts else (name or var_id)
            if vtype:
                detail += f" ({vtype})"
            if detail:
                var_details.append(detail)

            # Accumulate the process-stage score
            stage_num = _extract_stage_num(name or var_id)
            if stage_num is not None:
                try:
                    s = float(score) if score is not None else 0.0
                except (TypeError, ValueError):
                    s = 0.0
                stage_scores[stage_num] = stage_scores.get(stage_num, 0.0) + s

            # Find the highest-scoring variable
            if score is not None:
                try:
                    s = float(score)
                    if s > highest_score:
                        highest_score = s
                        highest_score_var = name or var_id
                except (TypeError, ValueError):
                    pass
    else:
        # --- Compatible with the old format: top10_vars + top10_details ---
        top10_vars = case.get("top10_vars") or case.get("candidate_vars") or []
        details = case.get("top10_details") or case.get("candidate_details") or []
        if details:
            for item in details[:10]:
                if isinstance(item, dict):
                    name = str(item.get("var") or item.get("variable") or item.get("name") or "")
                    vtype = str(item.get("type") or "")
                    desc = str(item.get("description") or item.get("semantic") or "")
                    score = item.get("score")
                else:
                    name = str(item)
                    vtype = ""
                    desc = ""
                    score = None
                if name:
                    var_names.append(name)
                if vtype:
                    var_types.append(vtype)
                if desc:
                    descriptions.append(desc)

                detail_parts = [name] if name else []
                detail = "/".join(detail_parts) if detail_parts else ""
                if vtype:
                    detail += f" ({vtype})"
                if detail:
                    var_details.append(detail)

                stage_num = _extract_stage_num(name)
                if stage_num is not None:
                    try:
                        s = float(score) if score is not None else 0.0
                    except (TypeError, ValueError):
                        s = 0.0
                    stage_scores[stage_num] = stage_scores.get(stage_num, 0.0) + s

                if score is not None:
                    try:
                        s = float(score)
                        if s > highest_score:
                            highest_score = s
                            highest_score_var = name
                    except (TypeError, ValueError):
                        pass
        else:
            var_names = [str(v) for v in top10_vars]

    # --- 2. Assemble the query parts ---
    # Variable details (ID/Name/Type) are the most core retrieval clues
    if var_details:
        parts.append(f"Top-{len(var_details)} 候选变量: " + "、".join(var_details))
    elif var_names:
        parts.append(f"候选变量: " + "、".join(var_names))

    # Highest-scoring variable - focus of the retrieval
    if highest_score_var and highest_score >= 0:
        parts.append(f"最高分根因变量: {highest_score_var} (score={highest_score:.2f})")

    # Variable types (e.g., state, continuous)
    if var_types:
        parts.append("变量类型: " + "、".join(sorted(set(var_types))))

    # Variable description/semantic information
    if descriptions:
        # Take the first 3 most representative descriptions to avoid an overly long query
        parts.append("变量描述: " + " | ".join(descriptions[:3]))

    # --- 3. Inject process-stage keywords ---
    # Pick the top-2 stages by cumulative score so retrieval focuses on the most likely faulty subsystem
    if stage_scores:
        top_stages = sorted(stage_scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
        stage_labels: List[str] = []
        for stage_num, _ in top_stages:
            labels = _STAGE_MAP.get(stage_num)
            if labels:
                stage_labels.extend(labels)
        if stage_labels:
            parts.append("工艺阶段: " + "、".join(stage_labels))

    # --- 4. Inject industrial-domain keywords ---
    # Infer the industrial domain comprehensively from variable names, descriptions, and types
    keywords = _infer_industrial_keywords(var_names, descriptions, var_types)
    if keywords:
        parts.append("工业关键词: " + "、".join(sorted(keywords)))

    # --- 5. Append a few case-related prompts to further focus the Embedding ---
    top_stage_nums = [s for s, _ in sorted(stage_scores.items(), key=lambda kv: kv[1], reverse=True)[:1]]
    primary_stage = top_stage_nums[0] if top_stage_nums else None
    focus = _build_focus_phrase(
        var_names=var_names,
        var_types=list(set(var_types)),
        primary_stage=primary_stage,
        highest_score_var=highest_score_var,
    )
    if focus:
        parts.append(focus)

    return " ".join(parts)


def _extract_stage_num(var_name: str) -> Optional[str]:
    """Parse the process-stage number (1-6) from a variable name such as P302, MV301, LIT401."""
    if not var_name:
        return None
    m = re.search(r"(\d+)$", var_name)
    if not m:
        return None
    num = m.group(1)
    # SWaT stage numbers are 1-6, matching the leading digit
    return num[0] if len(num) >= 1 and num[0] in _STAGE_MAP else None


def _build_focus_phrase(
    var_names: List[str],
    var_types: List[str],
    primary_stage: Optional[str],
    highest_score_var: str,
) -> str:
    """Build a single focus phrase from the case's main features, helping the Embedding lock onto relevant relationships."""
    chunks: List[str] = []

    if primary_stage and primary_stage in _STAGE_MAP:
        stage_labels = _STAGE_MAP[primary_stage]
        # Keep only the most representative Chinese/English labels
        chunks.append(f"{stage_labels[0]} ({stage_labels[1]}) 根因分析")

    type_set = set(t.lower() for t in var_types)
    if "state" in type_set:
        chunks.append("状态变量 root cause")
    if "continuous" in type_set:
        chunks.append("连续变量异常")

    if highest_score_var:
        chunks.append(f"{highest_score_var} 的关联变量关系")

    if not chunks:
        return ""
    return "聚焦: " + " / ".join(chunks)


def _infer_industrial_keywords(
    var_names: List[str], 
    descriptions: List[str] = None, 
    var_types: List[str] = None
) -> List[str]:
    """Comprehensively infer industrial-domain keywords from variable names, descriptions, and types.

    Matching priority:
    1. Variable-name prefix (e.g., P302 -> pump, MV301 -> valve, LIT401 -> level)
    2. Keywords in the description text (e.g., "Pump state" -> pump)
    3. Variable type (e.g., continuous -> level/pressure/flow sensor class)

    To keep the Embedding focused, only a few highly representative keywords are emitted.
    """
    if descriptions is None:
        descriptions = []
    if var_types is None:
        var_types = []

    keys: set = set()
    all_text = (
        " ".join(var_names).upper()
        + " "
        + " ".join(descriptions).lower()
        + " "
        + " ".join(var_types).lower()
    )

    # --- Infer from the variable-name prefix (priority, since it is most precise) ---
    # Match on "letter prefix + digit" to avoid startswith("P") misfiring on pressure type
    for v in var_names:
        name = (v or "").upper().strip()
        if not name:
            continue

        # Strip the trailing digit before judging the prefix (e.g., P302 -> P, MV301 -> MV, LIT401 -> LIT)
        m = re.match(r"^([A-Za-z]+)", name)
        prefix = m.group(1) if m else name

        if prefix == "FIT":
            keys.update(_KEYWORD_BANK["flow"])
        elif prefix == "LIT":
            keys.update(_KEYWORD_BANK["level"])
        elif prefix == "MV":
            keys.update(_KEYWORD_BANK["valve"])
        elif prefix == "P":
            # A plain P prefix denotes a pump (pump state/control)
            keys.update(_KEYWORD_BANK["pump"])
        elif prefix == "PIT":
            keys.update(_KEYWORD_BANK["pressure"])
        elif prefix == "DPIT":
            keys.update(_KEYWORD_BANK["pressure"])
            keys.update(_KEYWORD_BANK["membrane"])
        elif prefix == "AIT":
            keys.update(_KEYWORD_BANK["analyzer"])
        elif prefix == "UV":
            keys.update(_KEYWORD_BANK["uv"])

    # --- Fallback based on the description text (handles cases without a prefix hit) ---
    if "pump" in all_text or "泵" in all_text:
        keys.update(_KEYWORD_BANK["pump"])
    if "valve" in all_text or "阀门" in all_text:
        keys.update(_KEYWORD_BANK["valve"])
    if "flow" in all_text or "流量" in all_text:
        keys.update(_KEYWORD_BANK["flow"])
    if "level" in all_text or "液位" in all_text or "tank" in all_text:
        keys.update(_KEYWORD_BANK["level"])
    if "pressure" in all_text or "压力" in all_text:
        keys.update(_KEYWORD_BANK["pressure"])
    if "analyzer" in all_text or "ph" in all_text or "电导率" in all_text:
        keys.update(_KEYWORD_BANK["analyzer"])
    if "membrane" in all_text or "超滤" in all_text or "反渗透" in all_text:
        keys.update(_KEYWORD_BANK["membrane"])
    if "uv" in all_text or "紫外" in all_text:
        keys.update(_KEYWORD_BANK["uv"])

    # --- Supplement based on the variable type ---
    type_set = set(t.lower() for t in var_types)
    if "continuous" in type_set:
        # Continuous-type variables are usually sensors; add generic measurement terms
        keys.add("sensor")
        keys.add("传感器")
    if "state" in type_set:
        keys.add("state variable")
        keys.add("状态变量")

    # Limit the keyword count to avoid diluting the Embedding weight
    priority_order = [
        "pump", "valve", "flow", "level", "pressure",
        "membrane", "uv", "analyzer",
        "sensor", "传感器", "state variable", "状态变量",
    ]
    # Map the category names in _KEYWORD_BANK to the keywords above to keep representative terms
    selected: List[str] = []
    for cat in priority_order:
        if cat == "pump" and any(k in keys for k in _KEYWORD_BANK["pump"]):
            selected.extend(_KEYWORD_BANK["pump"][:3])
        elif cat == "valve" and any(k in keys for k in _KEYWORD_BANK["valve"]):
            selected.extend(_KEYWORD_BANK["valve"][:3])
        elif cat == "flow" and any(k in keys for k in _KEYWORD_BANK["flow"]):
            selected.extend(_KEYWORD_BANK["flow"][:3])
        elif cat == "level" and any(k in keys for k in _KEYWORD_BANK["level"]):
            selected.extend(_KEYWORD_BANK["level"][:3])
        elif cat == "pressure" and any(k in keys for k in _KEYWORD_BANK["pressure"]):
            selected.extend(_KEYWORD_BANK["pressure"][:3])
        elif cat == "membrane" and any(k in keys for k in _KEYWORD_BANK["membrane"]):
            selected.extend(_KEYWORD_BANK["membrane"][:3])
        elif cat == "uv" and any(k in keys for k in _KEYWORD_BANK["uv"]):
            selected.extend(_KEYWORD_BANK["uv"][:3])
        elif cat == "analyzer" and any(k in keys for k in _KEYWORD_BANK["analyzer"]):
            selected.extend(_KEYWORD_BANK["analyzer"][:3])
        elif cat in keys:
            selected.append(cat)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for k in selected:
        if k not in seen:
            seen.add(k)
            deduped.append(k)

    # If no inference matched, fall back to returning words that appear in all_text
    if not deduped:
        return sorted(keys)

    return deduped[:10]


class RAGRetriever:
    """Retrieve RAG context for a single case."""

    def __init__(
        self,
        rag_config: Dict[str, Any],
        embedder: RAGEmbedder,
        store: RAGVectorStore,
        logger: Optional[logging.Logger] = None,
    ):
        self.rag_config = rag_config
        self.embedder = embedder
        self.store = store
        self.top_k = int((rag_config.get("retrieval") or {}).get("top_k", 5))
        self.logger = logger or logging.getLogger("rag_retriever")

    @classmethod
    def from_config_path(cls, rag_config_path: str, logger: Optional[logging.Logger] = None) -> "RAGRetriever":
        cfg = read_yaml(rag_config_path)
        embedder = RAGEmbedder(cfg, logger=logger)
        store = RAGVectorStore(cfg, logger=logger)
        return cls(cfg, embedder, store, logger=logger)

    def retrieve(self, case: Dict[str, Any], top_k: Optional[int] = None) -> Dict[str, Any]:
        """Run RAG retrieval for a single case.

        Returns
        -------
        {
            "case_id": ...,
            "rag_query": "...",
            "retrieved_contexts": [
                {"content": ..., "source": ..., "score": 0.82, "metadata": {...}},
                ...
            ]
        }
        """
        case_id = case.get("case_id")
        query = build_rag_query(case)

        try:
            if not self.embedder.is_configured():
                raise RuntimeError(
                    f"Embedding 未配置（环境变量 '{self.embedder.api_key_env}' 未设置）。"
                )
            query_vec = self.embedder.embed_query(query)
            if not query_vec:
                raise RuntimeError("Embedding 返回了空向量。")
            result = self.store.query(query_embedding=query_vec, top_k=top_k or self.top_k)
            retrieved_contexts: List[Dict[str, Any]] = []
            for item in result.get("results", []):
                meta = item.get("metadata") or {}
                retrieved_contexts.append(
                    {
                        "content": item.get("content", ""),
                        "source": meta.get("source", ""),
                        "score": item.get("score"),
                        "metadata": meta,
                    }
                )
            return {
                "case_id": case_id,
                "rag_query": query,
                "retrieved_contexts": retrieved_contexts,
                "error": None,
            }
        except Exception as exc:
            self.logger.warning("[rag_retriever] case=%s 检索失败：%s", case_id, exc)
            return {
                "case_id": case_id,
                "rag_query": query,
                "retrieved_contexts": [],
                "error": str(exc),
            }
