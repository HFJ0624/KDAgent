"""Prompt builder: fill case content into the prompt template."""

import os
from typing import Any, Dict, List, Optional

from .utils import read_yaml


RAG_KNOWLEDGE_BLOCK_TEMPLATE = (
    "## 检索到的工业工艺知识\n\n"
    "以下知识片段检索自 {system_name} 工艺知识库。\n"
    "仅作为通用背景知识参考。\n"
    "请勿将其视为本案例的真实答案。\n\n"
    "{knowledge_items}\n"
    "提醒：最终根因必须仍从 Top-10 候选变量中选择，不得编造候选外变量。\n\n"
)


# All experimental conditions share the same output-length constraints. They only
# limit verbose expression and do not change the candidate, evidence, or root-cause
# judgment task; the goal is to reserve stable token space for the reasoning model's final JSON.
COMPACT_OUTPUT_RULES = """

## 最终输出长度约束（必须遵守）

请在内部完成分析后立即输出 JSON，不要在 JSON 前后输出计划、自言自语或重复任务说明。
只保留能够影响根因排序的关键证据，不要逐项复述全部 Top-10 或完整时间序列。

* reasoning：最多 180 个中文字符或 100 个英文单词。
* inference_process：固定 4 步；每步 analysis 最多 160 个中文字符或 90 个英文单词，每步最多引用 5 个 evidence_variables。
* numerical_evidence、temporal_evidence、type_aware_reasoning、process_relation_reasoning、why_not_other_candidates、uncertainty_analysis：每个字段最多 160 个中文字符或 90 个英文单词。
* 每个字段优先给出 1 至 3 个最关键事实；不同字段之间禁止重复同一句证据。
* 最终响应必须以字符 { 开始、以字符 } 结束，且只能包含一个合法 JSON 对象。
""".strip()


def format_rag_contexts(
    contexts: Optional[List[Dict[str, Any]]], system_name: str = "SWaT"
) -> str:
    """Convert the retrieved contexts into a text block; the default preserves the existing SWaT behavior."""
    if not contexts:
        return ""
    knowledge_lines: List[str] = []
    for idx, ctx in enumerate(contexts, start=1):
        content = (ctx.get("content") or "").strip()
        source = (ctx.get("source") or ctx.get("metadata", {}).get("source") or "未知")
        if not content:
            continue
        knowledge_lines.append(f"[知识 {idx}] (来源={source})\n{content}")
    if not knowledge_lines:
        return ""
    return RAG_KNOWLEDGE_BLOCK_TEMPLATE.format(
        system_name=system_name or "industrial system",
        knowledge_items="\n\n".join(knowledge_lines)
    )


def load_template(template_path: str) -> str:
    """Load the prompt template from a file; raises FileNotFoundError if it is missing.

    `prompts/rca_prompt_template.txt` is the single source of truth for the prompt,
    so a missing file is a hard configuration error instead of a silent fallback to a
    duplicated (and outdated) in-code template.
    """
    if not template_path:
        raise FileNotFoundError("No prompt-template path was provided")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def format_top10_candidates_compact(case: Dict[str, Any]) -> str:
    """Format the Top-10 candidate variable list into a compact summary string.
    
    Each variable keeps only its summary info, greatly reducing the prompt length:
    - var_id, name, type
    - rank, case_level_score
    - raw_start, raw_t0, raw_end
    - score_max, score_at_t0
    - residual_max_abs
    - first_high_score_time
    - short_trend_summary
    
    For example:
    V1 / LIT101 / continuous
    rank=10, score=2.9775
    raw: 512.25 -> 592.03 -> 620.03
    score_max=3.17, score_at_t0=1.72
    trend: level increases sharply around t=0
    """
    lines: List[str] = []
    details = case.get("top10_details") or case.get("candidate_details") or []
    top10_vars = case.get("top10_vars") or case.get("candidate_vars") or []
    top_k_vars = case.get("top_k_variables") or []

    if top_k_vars and isinstance(top_k_vars, list):
        for idx, item in enumerate(top_k_vars[:10], start=1):
            if not isinstance(item, dict):
                lines.append(f"{idx}. {item}")
                continue
            
            var_id = str(item.get("var_id", "") or "")
            name = str(item.get("name", "") or "")
            vtype = str(item.get("type", "") or "")
            rank = item.get("rank")
            score = item.get("case_level_score")
            
            header = f"{idx}. {var_id}"
            if name and name != var_id:
                header += f" / {name}"
            if vtype:
                header += f" / {vtype}"
            lines.append(header)
            
            meta_parts = []
            if rank is not None:
                meta_parts.append(f"rank={rank}")
            if score is not None:
                try:
                    meta_parts.append(f"score={float(score):.4f}")
                except (TypeError, ValueError):
                    meta_parts.append(f"score={score}")
            if meta_parts:
                lines.append(f"   {', '.join(meta_parts)}")
            
            ts = item.get("time_series") or []
            if isinstance(ts, list) and ts:
                raw_values = []
                score_values = []
                for step in ts:
                    if not isinstance(step, dict):
                        continue
                    raw_v = step.get("raw_value", step.get("raw"))
                    score_v = step.get("final_score", step.get("score"))
                    if raw_v is not None and raw_v != "":
                        try:
                            raw_values.append(f"{float(raw_v):.2f}")
                        except (TypeError, ValueError):
                            raw_values.append(str(raw_v))
                    if score_v is not None and score_v != "":
                        try:
                            score_values.append(float(score_v))
                        except (TypeError, ValueError):
                            pass
                
                if raw_values:
                    if len(raw_values) >= 3:
                        lines.append(f"   raw: {raw_values[0]} -> {raw_values[len(raw_values)//2]} -> {raw_values[-1]}")
                    else:
                        lines.append(f"   raw: {' -> '.join(raw_values)}")
                
                if score_values:
                    score_max = max(score_values)
                    score_t0 = score_values[len(score_values)//2] if len(score_values) > 1 else score_values[0]
                    lines.append(f"   score_max={score_max:.2f}, score_at_t0={score_t0:.2f}")
            
            desc = str(item.get("description", "") or "")
            if desc:
                lines.append(f"   trend: {desc[:100]}")
    
    elif details:
        for idx, item in enumerate(details[:10], start=1):
            if not isinstance(item, dict):
                lines.append(f"{idx}. {item}")
                continue
            
            var_id = str(item.get("var") or item.get("var_id") or item.get("variable") or item.get("name") or f"V{idx}")
            name = str(item.get("name") or "")
            vtype = str(item.get("type", "") or "")
            rank = item.get("rank")
            score = item.get("score")
            if score is None:
                score = item.get("case_level_score")
            
            header = f"{idx}. {var_id}"
            if name and name != var_id:
                header += f" / {name}"
            if vtype:
                header += f" / {vtype}"
            lines.append(header)
            
            meta_parts = []
            if rank is not None:
                meta_parts.append(f"rank={rank}")
            if score is not None:
                try:
                    meta_parts.append(f"score={float(score):.4f}")
                except (TypeError, ValueError):
                    meta_parts.append(f"score={score}")
            if meta_parts:
                lines.append(f"   {', '.join(meta_parts)}")
            
            raw = item.get("raw", "")
            if raw:
                lines.append(f"   raw: {raw}")
            
            desc = str(item.get("description") or item.get("semantic") or "")
            if desc:
                lines.append(f"   trend: {desc[:100]}")
    
    else:
        for idx, var in enumerate(top10_vars[:10], start=1):
            lines.append(f"{idx}. {var}")
    
    return "\n".join(lines)


def format_top10_candidates(case: Dict[str, Any]) -> str:
    """Format the Top-10 candidate variable list into a readable string.

    Supports two structures:
      1. `top_k_variables` (new format): each variable has var_id/name/type/case_level_score/description/time_series
      2. `top10_details` (compat format): each variable has var/name/type/score/raw/recon/residual/semantic
    """
    lines: List[str] = []
    details = case.get("top10_details") or case.get("candidate_details") or []
    top10_vars = case.get("top10_vars") or case.get("candidate_vars") or []

    if details:
        for idx, item in enumerate(details[:10], start=1):
            if not isinstance(item, dict):
                lines.append(f"{idx}. 变量：{item}")
                continue

            var_id = item.get("var") or item.get("var_id") or item.get("variable") or item.get("name") or f"V{idx}"
            name = item.get("name") or ""
            vtype = item.get("type", "")
            score = item.get("score")
            if score is None:
                score = item.get("case_level_score")
            raw = item.get("raw", "")
            recon = item.get("recon", "")
            residual = item.get("residual", "")
            description = item.get("description") or item.get("semantic") or ""
            ts = item.get("time_series") or []

            header = f"{idx}. 变量：{var_id}"
            if name and name != var_id:
                header += f" / {name}"
            if vtype:
                header += f" | 类型：{vtype}"
            if score is not None and score != "":
                try:
                    header += f" | 分数：{float(score):.4f}"
                except (TypeError, ValueError):
                    header += f" | 分数：{score}"
            if raw != "" and raw is not None:
                header += f" | 原始：{raw}"
            if recon != "" and recon is not None:
                header += f" | 重构：{recon}"
            if residual != "" and residual is not None:
                header += f" | 残差：{residual}"

            lines.append(header)
            if description:
                lines.append(f"   描述：{description}")

            # Time-series summary: by default show only the last few key points (to avoid an overly long prompt)
            if isinstance(ts, list) and ts:
                sample = ts  # keep all; let the LLM decide
                lines.append("   时间序列证据：")
                ts_parts = []
                for step in sample:
                    if not isinstance(step, dict):
                        continue
                    dt = step.get("time_offset", step.get("dt", ""))
                    raw_v = step.get("raw_value", step.get("raw", ""))
                    recon_v = step.get("recon_value", step.get("recon", ""))
                    res_v = step.get("decoder_residual", step.get("residual", ""))
                    score_v = step.get("final_score", step.get("score", ""))
                    ts_parts.append(
                        f"(dt={dt}, raw={raw_v}, recon={recon_v}, res={res_v}, score={score_v})"
                    )
                if ts_parts:
                    # By default show at most 12 time steps to avoid an overly long prompt
                    shown = ts_parts[:12]
                    lines.append("   " + " ".join(shown))
                    if len(ts_parts) > len(shown):
                        lines.append(f"   ... (省略 {len(ts_parts) - len(shown)} 个时间步)")
    else:
        for idx, var in enumerate(top10_vars[:10], start=1):
            lines.append(f"{idx}. 变量：{var}")

    return "\n".join(lines)


def format_public_numerical_evidence(
    case: Dict[str, Any], compact_evidence: bool = False
) -> str:
    """Generate the online numerical-evidence view shared by all methods.

    This entry only reads the normalized candidates and their time-series fields; it
    never reads ground truth, attack labels, or scoring fields. The main experiment
    uses the full format by default; its time series still follows the existing
    entry's 12-point display limit, so the newly fixed policy is not mistaken for
    historical input.
    """
    return (
        format_top10_candidates_compact(case)
        if compact_evidence
        else format_top10_candidates(case)
    )


def validate_rag_template_support(
    template: str,
    rag_contexts: Optional[List[Dict[str, Any]]],
) -> None:
    """Prevent the silent data-loss path where retrieval succeeds but the template has no slot."""
    has_nonempty_context = any(
        isinstance(context, dict) and str(context.get("content") or "").strip()
        for context in (rag_contexts or [])
    )
    if has_nonempty_context and "{rag_knowledge_block}" not in template:
        raise ValueError("rag_template_missing_placeholder")


def build_prompt(
    case: Dict[str, Any],
    template: str,
    process_knowledge: str = "",
    rag_contexts: Optional[List[Dict[str, Any]]] = None,
    compact_evidence: bool = False,
) -> str:
    """Fill the case content into the template and build the final prompt string.
    
    Args:
        case: The case data dict
        template: The prompt template string
        process_knowledge: The process-knowledge text
        rag_contexts: The context list retrieved via RAG
        compact_evidence: Whether to use compact evidence mode, greatly reducing prompt length
    """
    # First check RAG injection capability, then handle the case's own prompt, so the
    # full-prompt path does not silently drop the retrieval results.
    validate_rag_template_support(template, rag_contexts)

    # If the case already contains a complete prompt (no placeholders), use it directly
    if case.get("prompt") and "{process_knowledge}" not in template and "{top10_candidates}" not in template and "{rag_knowledge_block}" not in template:
        return case["prompt"]

    top10_text = format_public_numerical_evidence(case, compact_evidence=compact_evidence)
    rag_block = format_rag_contexts(
        rag_contexts,
        system_name=str(case.get("system_name") or "SWaT"),
    )
    prompt = (
        template.replace("{process_knowledge}", process_knowledge)
        .replace("{top10_candidates}", top10_text)
        .replace("{rag_knowledge_block}", rag_block)
    )
    if "## 最终输出长度约束（必须遵守）" not in prompt:
        prompt = f"{prompt.rstrip()}\n\n{COMPACT_OUTPUT_RULES}\n"
    return prompt


def load_process_knowledge(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""
