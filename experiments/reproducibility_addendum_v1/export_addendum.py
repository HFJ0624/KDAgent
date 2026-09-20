"""Read-only consolidation of the existing KDAgent, Serial, and ReAct reproducibility materials."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPORT_ROOT = OUTPUT_ROOT / "KDAgent_reproducibility_addendum"
ZIP_PATH = OUTPUT_ROOT / "KDAgent_reproducibility_addendum.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source_relative: str, target_relative: str, copied: List[Dict[str, Any]]) -> None:
    source = PROJECT_ROOT / source_relative
    target = EXPORT_ROOT / target_relative
    if not source.is_file():
        raise FileNotFoundError(f"待导出文件不存在：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = sha256(source)
    target_hash = sha256(target)
    if source_hash != target_hash:
        raise RuntimeError(f"复制后哈希不一致：{source}")
    copied.append(
        {
            "package_path": target_relative.replace("\\", "/"),
            "source_path": str(source),
            "source_experiment": experiment_for(source_relative),
            "file_role": role_for(target_relative),
            "size_bytes": target.stat().st_size,
            "sha256": target_hash,
            "generated_or_copied": "copied_without_modification",
        }
    )


def experiment_for(path: str) -> str:
    normalized = path.replace("\\", "/")
    if "react_adapted_v1/comparators/wadi/kdagent" in normalized:
        return "WADI KDAgent: run 1 reused from 2026-09-03; runs 2-3 generated 2026-09-12"
    if "react_adapted_v1/comparators/wadi/serial" in normalized:
        return "WADI Serial: runs 1-3 generated 2026-09-12"
    if "final_experiments_frozen_v2/dual_branch" in normalized:
        return "SWaT KDAgent historical run, 20 episodes x 3 runs, 2026-07-23"
    if "final_experiments_frozen_v1/rag_self" in normalized:
        return "SWaT Serial historical run, 20 episodes x 3 runs, 2026-07-23"
    if normalized.startswith("outputs/react_adapted_v1/react/"):
        return "ReAct formal run, SWaT 60 + WADI 39 records, 2026-09-12"
    if normalized.startswith("outputs/react_adapted_v1/"):
        return "ReAct v1 combined offline analysis/output"
    if "wadi_external_validation_v1" in normalized:
        return "WADI external-validation preparation or frozen input"
    if "swat_s2s_raw_window_fixed" in normalized or "episode_audit_v1" in normalized:
        return "SWaT frozen benchmark input or episode audit"
    return "shared implementation/configuration"


def role_for(path: str) -> str:
    normalized = path.replace("\\", "/")
    if "/logs/" in normalized:
        return "historical_run_log"
    if "/metrics/" in normalized or normalized.endswith("metrics.json"):
        return "historical_metrics"
    if normalized.endswith("raw_model_calls.jsonl"):
        return "react_raw_model_calls"
    if normalized.endswith("tool_events.jsonl"):
        return "react_tool_events"
    if normalized.endswith("scoring_keys.jsonl"):
        return "offline_scoring_labels_separated_from_online_context"
    if normalized.endswith("final_records.jsonl"):
        return "react_final_diagnosis_records"
    if normalized.endswith((".py", ".txt")) and "implementation" in normalized:
        return "implementation_or_prompt"
    if "knowledge" in normalized:
        return "knowledge_source_or_configuration"
    if "events" in normalized:
        return "frozen_event_input_or_manifest"
    return "experiment_evidence"


def files_to_copy() -> List[tuple[str, str]]:
    react_impl = [
        "experiments/react_adapted_v1/__init__.py",
        "experiments/react_adapted_v1/react_agent.py",
        "experiments/react_adapted_v1/run_experiment.py",
        "experiments/react_adapted_v1/test_react_adapted.py",
        "experiments/react_adapted_v1/config.yaml",
        "experiments/react_adapted_v1/experiment_protocol.md",
        "experiments/react_adapted_v1/README.md",
        "experiments/react_adapted_v1/prompts/react_system.txt",
    ]
    react_outputs = [
        "outputs/react_adapted_v1/react/raw_model_calls.jsonl",
        "outputs/react_adapted_v1/react/tool_events.jsonl",
        "outputs/react_adapted_v1/react/final_records.jsonl",
        "outputs/react_adapted_v1/react/scoring_keys.jsonl",
        "outputs/react_adapted_v1/record_level_results.csv",
        "outputs/react_adapted_v1/summary_metrics.csv",
        "outputs/react_adapted_v1/paired_comparisons.csv",
        "outputs/react_adapted_v1/paired_episode_differences.csv",
        "outputs/react_adapted_v1/budget_and_failure_audit.csv",
        "outputs/react_adapted_v1/formal_run_status.json",
        "outputs/react_adapted_v1/reproducibility_manifest.json",
        "outputs/react_adapted_v1/data_availability_report.json",
        "outputs/react_adapted_v1/data_availability_report.md",
        "outputs/react_adapted_v1/kb_label_isolation_audit.csv",
        "outputs/react_adapted_v1/overlap_group_sensitivity.csv",
        "outputs/react_adapted_v1/dry_run_budget.json",
        "outputs/react_adapted_v1/paper_ready_table.csv",
        "outputs/react_adapted_v1/results_summary_zh.md",
    ]
    shared_impl = [
        "src/main.py",
        "src/dual_branch_fusion_agent.py",
        "src/iterative_agent.py",
        "src/response_validator.py",
        "src/response_parser.py",
        "src/model_client.py",
        "src/data_loader.py",
        "src/prompt_builder.py",
        "src/rag_retriever.py",
        "src/rag_embedder.py",
        "src/rag_vector_store.py",
        "src/rag_kb_builder.py",
        "src/build_rag_index.py",
        "src/evaluator.py",
        "src/utils.py",
        "configs/models.yaml",
        "configs/rag.yaml",
        "prompts/rca_prompt_template.txt",
        "experiments/wadi_external_validation_v1/wadi_adapter.py",
        "experiments/wadi_external_validation_v1/prepare_wadi.py",
        "experiments/wadi_external_validation_v1/run_wadi_external_validation.py",
        "experiments/wadi_external_validation_v1/configs/rag_wadi.yaml",
        "experiments/wadi_external_validation_v1/prompts/wadi_rca_prompt_template.txt",
        "experiments/wadi_external_validation_v1/README.md",
    ]
    historical = [
        "outputs/final_experiments_frozen_v2/dual_branch_fusion_agent_deepseek-v4-pro/logs/rca_experiment_deepseek-v4-pro_with-rag_20260723_132325.log",
        "outputs/final_experiments_frozen_v2/dual_branch_fusion_agent_deepseek-v4-pro/metrics/deepseek-v4-pro_metrics.json",
        "outputs/final_experiments_frozen_v1/rag_self_refinement_agent_deepseek-v4-pro/logs/rca_experiment_deepseek-v4-pro_with-rag_20260723_005609.log",
        "outputs/final_experiments_frozen_v1/rag_self_refinement_agent_deepseek-v4-pro/metrics/deepseek-v4-pro_metrics.json",
        "outputs/react_adapted_v1/comparators/wadi/kdagent_deepseek-v4-pro/logs/rca_experiment_deepseek-v4-pro_with-rag_20260912_163818.log",
        "outputs/react_adapted_v1/comparators/wadi/kdagent_deepseek-v4-pro/metrics/deepseek-v4-pro_metrics.json",
        "outputs/react_adapted_v1/comparators/wadi/serial_deepseek-v4-pro/logs/rca_experiment_deepseek-v4-pro_with-rag_20260912_174112.log",
        "outputs/react_adapted_v1/comparators/wadi/serial_deepseek-v4-pro/metrics/deepseek-v4-pro_metrics.json",
    ]
    event_files = [
        "data/swat_s2s_raw_window_fixed/llm_prompt_cases.jsonl",
        "data/swat_s2s_raw_window_fixed/case_top10_summary.csv",
        "data/swat_s2s_raw_window_fixed/export_summary.json",
        "data/swat_s2s_raw_window_fixed/variables_meta.csv",
        "analysis/episode_audit_v1/swat_episode_audit.csv",
        "outputs/wadi_external_validation_v1/prepared_data/llm_prompt_cases.jsonl",
        "outputs/wadi_external_validation_v1/prepared_data/wadi_episode_manifest.csv",
        "outputs/wadi_external_validation_v1/prepared_data/wadi_episode_manifest.json",
        "outputs/wadi_external_validation_v1/prepared_data/wadi_tarca_top10.json",
        "outputs/wadi_external_validation_v1/prepared_data/export_summary.json",
        "outputs/wadi_external_validation_v1/prepared_data/variables_meta.csv",
        "outputs/wadi_external_validation_v1/kb_build_manifest.json",
    ]

    mappings: List[tuple[str, str]] = []
    for source in react_impl:
        relative = source.removeprefix("experiments/react_adapted_v1/")
        mappings.append((source, f"react/implementation/{relative}"))
    for source in react_outputs:
        relative = source.removeprefix("outputs/react_adapted_v1/")
        destination = "react/run_records" if relative.startswith("react/") else "react/results"
        mappings.append((source, f"{destination}/{relative.removeprefix('react/')}"))
    for source in shared_impl:
        mappings.append((source, f"methods/shared_implementation/{source}"))
    for source in historical:
        normalized = source.replace("\\", "/")
        if "dual_branch_fusion" in normalized:
            name = "swat_kdagent"
        elif "final_experiments_frozen_v1" in normalized:
            name = "swat_serial"
        elif "/wadi/kdagent_" in normalized:
            name = "wadi_kdagent"
        else:
            name = "wadi_serial"
        mappings.append((source, f"methods/historical_evidence/{name}/{Path(source).name}"))
    for source in event_files:
        dataset = "wadi" if "wadi" in source.lower() else "swat"
        mappings.append((source, f"events/{dataset}/{Path(source).name}"))

    for path in sorted((PROJECT_ROOT / "data" / "rag_kb").glob("*.md")):
        source = str(path.relative_to(PROJECT_ROOT))
        mappings.append((source, f"knowledge/swat/source_documents/{path.name}"))
    mappings.append(("data/swat_s2s_raw_window_fixed/process_knowledge_template.md", "knowledge/swat/source_documents/process_knowledge_template.md"))
    mappings.append(("data/swat_s2s_raw_window_fixed/variables_meta.csv", "knowledge/swat/source_documents/variables_meta.csv"))
    mappings.append(("configs/rag.yaml", "knowledge/swat/index_config/rag.yaml"))
    for path in sorted((PROJECT_ROOT / "outputs" / "wadi_external_validation_v1" / "prepared_data" / "rag_kb").glob("*.md")):
        source = str(path.relative_to(PROJECT_ROOT))
        mappings.append((source, f"knowledge/wadi/source_documents/{path.name}"))
    mappings.append(("outputs/wadi_external_validation_v1/prepared_data/process_knowledge_template.md", "knowledge/wadi/source_documents/process_knowledge_template.md"))
    mappings.append(("outputs/wadi_external_validation_v1/prepared_data/variables_meta.csv", "knowledge/wadi/source_documents/variables_meta.csv"))
    mappings.append(("experiments/wadi_external_validation_v1/configs/rag_wadi.yaml", "knowledge/wadi/index_config/rag_wadi.yaml"))
    return mappings


def source_document_path(dataset: str, source: str) -> Path:
    if dataset == "swat":
        if source == "variables_meta.csv":
            return PROJECT_ROOT / "data" / "swat_s2s_raw_window_fixed" / source
        if source == "process_knowledge_template.md":
            return PROJECT_ROOT / "data" / "swat_s2s_raw_window_fixed" / source
        return PROJECT_ROOT / "data" / "rag_kb" / source
    base = PROJECT_ROOT / "outputs" / "wadi_external_validation_v1" / "prepared_data"
    if source in {"variables_meta.csv", "process_knowledge_template.md"}:
        return base / source
    return base / "rag_kb" / source


def source_classification(source: str) -> tuple[str, str]:
    if source == "variables_meta.csv":
        return "variable_metadata", "逐变量生成；不属于人工规则。变量元数据的上游生成过程见数据准备代码。"
    if source == "process_knowledge_template.md":
        return "process_template", "工艺模板；仓库未保存逐段人工编辑历史。"
    if source.startswith("kb_official_swat_"):
        return "external_source_summary", "带来源 URL/DOI 的外部资料摘要；不是原始外部文档全文。"
    return "generic_rule_or_relation", "仓库内通用规则/关系文件；未保存具体作者与人工核对历史。"


def chunk_position(chunk_id: str, metadata: Mapping[str, Any]) -> str:
    if metadata.get("var_index") not in (None, ""):
        return f"var_index={metadata['var_index']}"
    match = re.search(r"__(\d+)$", chunk_id)
    return f"markdown_chunk_ordinal={int(match.group(1))}" if match else "未保存精确位置"


def export_knowledge_chunks() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("导出实际知识块需要读取现有 ChromaDB，但当前环境缺少 chromadb。") from exc

    specs = [
        ("swat", PROJECT_ROOT / "outputs" / "chroma_db", "swat_process_kb"),
        ("wadi", PROJECT_ROOT / "outputs" / "wadi_external_validation_v1" / "chroma_db", "wadi_process_kb_v1"),
    ]
    chunk_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    for dataset, persist, collection_name in specs:
        collection = chromadb.PersistentClient(path=str(persist)).get_collection(collection_name)
        result = collection.get(include=["documents", "metadatas"])
        rows = []
        for chunk_id, content, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
            metadata = metadata or {}
            source = str(metadata.get("source") or "")
            category, manual_note = source_classification(source)
            row = {
                "dataset": dataset,
                "collection_name": collection_name,
                "chunk_id": chunk_id,
                "source": source,
                "source_position": chunk_position(chunk_id, metadata),
                "doc_type": metadata.get("doc_type", ""),
                "stage": metadata.get("stage", ""),
                "var_id": metadata.get("var_id", ""),
                "variable_name": metadata.get("variable_name", ""),
                "source_category": category,
                "manual_template_or_rule_note": manual_note,
                "source_title": metadata.get("source_title", ""),
                "source_url": metadata.get("source_url", ""),
                "source_doi": metadata.get("source_doi", ""),
                "source_publisher": metadata.get("source_publisher", ""),
                "source_authors": metadata.get("source_authors", ""),
                "source_year": metadata.get("source_year", ""),
                "source_accessed": metadata.get("source_accessed", ""),
                "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                "content": content or "",
            }
            rows.append(row)
            chunk_rows.append(row)

        counts = Counter(row["source"] for row in rows)
        first_by_source = {row["source"]: row for row in rows}
        for source, count in sorted(counts.items()):
            first = first_by_source[source]
            source_path = source_document_path(dataset, source)
            source_rows.append(
                {
                    "dataset": dataset,
                    "collection_name": collection_name,
                    "source": source,
                    "chunk_count": count,
                    "source_category": first["source_category"],
                    "manual_template_or_rule_note": first["manual_template_or_rule_note"],
                    "source_file": str(source_path),
                    "source_file_sha256": sha256(source_path) if source_path.exists() else "",
                    "source_file_last_write_time": datetime.fromtimestamp(source_path.stat().st_mtime).isoformat() if source_path.exists() else "",
                    "source_title": first["source_title"],
                    "source_url": first["source_url"],
                    "source_doi": first["source_doi"],
                    "source_accessed": first["source_accessed"],
                    "exact_character_offsets_preserved": "no",
                }
            )

    chunk_fields = [
        "dataset", "collection_name", "chunk_id", "source", "source_position", "doc_type", "stage",
        "var_id", "variable_name", "source_category", "manual_template_or_rule_note", "source_title",
        "source_url", "source_doi", "source_publisher", "source_authors", "source_year", "source_accessed",
        "metadata_json", "content",
    ]
    source_fields = [
        "dataset", "collection_name", "source", "chunk_count", "source_category",
        "manual_template_or_rule_note", "source_file", "source_file_sha256", "source_file_last_write_time",
        "source_title", "source_url", "source_doi", "source_accessed", "exact_character_offsets_preserved",
    ]
    write_csv(EXPORT_ROOT / "knowledge" / "knowledge_chunks.csv", sorted(chunk_rows, key=lambda row: (row["dataset"], row["chunk_id"])), chunk_fields)
    write_csv(EXPORT_ROOT / "knowledge" / "knowledge_source_catalog.csv", source_rows, source_fields)
    return chunk_rows, source_rows


def export_wadi_episode_table() -> None:
    source = PROJECT_ROOT / "outputs" / "wadi_external_validation_v1" / "prepared_data" / "wadi_episode_manifest.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    fields = [
        "episode_id", "source_attack_id", "attack_row_start", "attack_row_end", "declared_start_time",
        "declared_end_time", "root_tags", "gt_vars", "gt_names", "label_type", "included",
        "selection_reason", "exclusion_reason", "tarca_top10", "candidate_covered", "best_gt_rank",
    ]
    write_csv(EXPORT_ROOT / "events" / "wadi" / "wadi_episode_reproducibility_table.csv", records, fields)


def configuration_rows() -> List[Dict[str, str]]:
    swat_k_log = "methods/historical_evidence/swat_kdagent/rca_experiment_deepseek-v4-pro_with-rag_20260723_132325.log"
    swat_s_log = "methods/historical_evidence/swat_serial/rca_experiment_deepseek-v4-pro_with-rag_20260723_005609.log"
    wadi_k_log = "methods/historical_evidence/wadi_kdagent/rca_experiment_deepseek-v4-pro_with-rag_20260912_163818.log"
    wadi_s_log = "methods/historical_evidence/wadi_serial/rca_experiment_deepseek-v4-pro_with-rag_20260912_174112.log"
    code = "methods/shared_implementation/"
    react = "react/implementation/"
    rows: List[Dict[str, str]] = []

    def add(method: str, dataset: str, setting: str, value: str, evidence_class: str, evidence_file: str, note: str = "") -> None:
        rows.append({"method": method, "dataset": dataset, "setting": setting, "value": value, "evidence_class": evidence_class, "evidence_file": evidence_file, "notes": note})

    for method, dataset, log in (
        ("KDAgent", "SWaT", swat_k_log), ("KDAgent", "WADI", wadi_k_log),
        ("Serial", "SWaT", swat_s_log), ("Serial", "WADI", wadi_s_log),
    ):
        add(method, dataset, "model", "deepseek-v4-pro / openai_compatible", "historical_run_log", log)
        add(method, dataset, "temperature", "0.2", "historical_run_log", log)
        add(method, dataset, "max_tokens_per_generation", "8192", "historical_run_log", log)
        add(method, dataset, "thinking_budget", "2048", "historical_run_log", log)
        add(method, dataset, "generation_timeout", "120 seconds", "historical_run_log", log)
        add(method, dataset, "top_level_max_retries", "5", "historical_run_log", log, "这是日志中的客户端配置，不等同于 Agent 内部每次调用的显式覆盖值。")
        add(method, dataset, "retrieval_top_k", "5", "historical_run_log", log)
        add(method, dataset, "max_iterations", "3", "historical_run_log", log)

    for method in ("KDAgent", "Serial"):
        add(method, "both", "effective_generation_network_attempt_cap", "3 per logical model call", "current_implementation", f"{code}src/iterative_agent.py", "历史日志未单列内部覆盖值；当前 run_case 显式传入 max_retries=3。")
        add(method, "both", "embedding_retry_cap", "5", "current_code_default", f"{code}src/rag_embedder.py", "rag.yaml 未显式配置 max_retries，因此当前代码默认 5；历史实际失败次数未单独保存。")
        add(method, "both", "embedding_timeout", "60 seconds", "current_code_default", f"{code}src/rag_embedder.py", "代码读取 request_timeout_s；现有 rag.yaml 写的是 timeout，因此按当前实现回落到 60。历史请求耗时上限未独立记录。")
        add(method, "both", "autonomous_tools", "none", "current_implementation", f"{code}src/main.py", "检索由外层编排在生成前执行，不由模型自主选择。")
        add(method, "both", "short_ranking_handling", "accept 1-5 valid candidates; filter illegal/duplicates; no padding", "current_implementation", f"{code}src/response_validator.py; {code}src/response_parser.py")

    add("KDAgent", "both", "input_information", "Evidence branch: frozen Top-10 evidence + process template, no retrieved chunks. Retrieval branch: same evidence/template + pre-retrieved Top-5 chunks.", "current_implementation", f"{code}src/main.py; {code}src/prompt_builder.py")
    add("KDAgent", "both", "retrieval_scope", "dataset-specific Chroma collection; one case-derived retrieval before branch generation", "historical_log_and_code", f"{code}src/main.py; knowledge/swat/index_config/rag.yaml; knowledge/wadi/index_config/rag_wadi.yaml")
    add("KDAgent", "both", "maximum_logical_generations", "4 (Evidence up to 3 + Retrieval branch 1)", "current_implementation", f"{code}src/dual_branch_fusion_agent.py; {code}src/iterative_agent.py")
    add("KDAgent", "both", "stopping_and_fallback", "Evidence stops on validator pass; final evidence round is forced-choice; exhausted evidence returns UNKNOWN. Fusion uses evidence primary, otherwise retrieval primary, otherwise UNKNOWN.", "current_implementation", f"{code}src/iterative_agent.py; {code}src/dual_branch_fusion_agent.py")

    add("Serial", "both", "input_information", "Frozen Top-10 evidence + process template + pre-retrieved Top-5 chunks in one iterative context.", "current_implementation", f"{code}src/main.py; {code}src/prompt_builder.py")
    add("Serial", "both", "retrieval_scope", "dataset-specific Chroma collection; one case-derived retrieval before generation", "historical_log_and_code", f"{code}src/main.py; knowledge/swat/index_config/rag.yaml; knowledge/wadi/index_config/rag_wadi.yaml")
    add("Serial", "both", "maximum_logical_generations", "3", "historical_log_and_code", f"{swat_s_log}; {wadi_s_log}; {code}src/iterative_agent.py")
    add("Serial", "both", "stopping_and_fallback", "Stop on validator pass; round 2 uses targeted refinement; round 3 uses forced-choice; all failed returns UNKNOWN with empty ranking.", "current_implementation", f"{code}src/iterative_agent.py")

    protocol = "react/implementation/experiment_protocol.md"
    raw = "react/run_records/raw_model_calls.jsonl"
    tool = "react/run_records/tool_events.jsonl"
    for dataset in ("SWaT", "WADI"):
        add("ReAct", dataset, "model", "deepseek-v4-pro / openai_compatible", "formal_protocol_and_raw_records", f"{protocol}; {raw}")
        add("ReAct", dataset, "temperature", "0.2", "formal_protocol_hash_linked_to_records", f"{protocol}; react/run_records/final_records.jsonl")
        add("ReAct", dataset, "max_tokens_per_generation", "8192", "formal_protocol_hash_linked_to_records", protocol)
        add("ReAct", dataset, "thinking_budget", "2048", "formal_protocol_hash_linked_to_records", f"{protocol}; {raw}", "raw records保存 provider 报告的 reasoning_tokens，不保存服务端内部模型修订号。")
        add("ReAct", dataset, "generation_timeout", "120 seconds", "formal_protocol_hash_linked_to_records", protocol)
        add("ReAct", dataset, "generation_network_attempt_cap", "5", "formal_protocol_and_current_config", f"{protocol}; react/implementation/config.yaml")
        add("ReAct", dataset, "retrieval_top_k", "5 per search", "formal_protocol_and_tool_records", f"{protocol}; {tool}")
        add("ReAct", dataset, "maximum_logical_generations_and_tool_actions", "4", "formal_protocol_and_raw_records", f"{protocol}; {raw}; {tool}")
    add("ReAct", "both", "input_information", "Initial context: dataset, episode ID, frozen candidate identity metadata and tool schemas. Evidence/knowledge enter only through executed tool observations.", "implementation_and_raw_records", f"{react}react_agent.py; {raw}; {tool}")
    add("ReAct", "both", "autonomous_tools", "get_episode_evidence; search_domain_knowledge; submit_ranking", "implementation_and_tool_records", f"{react}react_agent.py; {tool}")
    add("ReAct", "both", "retrieval_scope", "model-written query against the corresponding dataset-specific Chroma collection; repeat search allowed", "implementation_and_tool_records", f"{react}react_agent.py; {tool}")
    add("ReAct", "both", "stopping_and_fallback", "Stop on valid submit_ranking; otherwise stop on API failure, completion budget, or four calls. Failure keeps an empty prediction; no numerical Top-1 padding.", "current_implementation", f"{react}react_agent.py")
    add("ReAct", "both", "short_ranking_handling", "accept 1-5 unique legal candidates with primary equal to Rank 1; no padding", "current_implementation", f"{react}react_agent.py")

    add("KDAgent", "SWaT", "result_provenance", "existing historical results: 60 records from 2026-07-23", "historical_outputs", swat_k_log)
    add("Serial", "SWaT", "result_provenance", "existing historical results: 60 records from 2026-07-23", "historical_outputs", swat_s_log)
    add("KDAgent", "WADI", "result_provenance", "run 1 reused from 2026-09-03; runs 2-3 generated 2026-09-12", "runner_and_historical_outputs", f"{react}run_experiment.py; {wadi_k_log}")
    add("Serial", "WADI", "result_provenance", "runs 1-3 generated 2026-09-12", "historical_outputs", wadi_s_log)
    add("ReAct", "both", "result_provenance", "99 formal records generated 2026-09-12 (SWaT 60, WADI 39)", "formal_outputs", "react/results/formal_run_status.json; react/run_records/final_records.jsonl")
    return rows


def write_configuration_comparison(rows: List[Dict[str, str]]) -> None:
    fields = ["method", "dataset", "setting", "value", "evidence_class", "evidence_file", "notes"]
    write_csv(EXPORT_ROOT / "configuration_comparison.csv", rows, fields)
    md = [
        "# 三种方法配置对照",
        "",
        "本表严格区分历史日志、正式记录关联协议、当前实现和当前代码默认值。`current_implementation` 或 `current_code_default` 不自动等同于未记录的历史事实。",
        "",
        "| Method | Dataset | Setting | Value | Evidence class | Evidence | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        values = [row[field].replace("|", "\\|").replace("\n", " ") for field in fields]
        md.append("| " + " | ".join(values) + " |")
    write_text(EXPORT_ROOT / "configuration_comparison.md", "\n".join(md))


def write_run_command_evidence() -> None:
    text = """# 运行命令与版本依据

## ReAct

仓库保存的 README/协议规定以下正式命令（密钥值已替换）：

```powershell
$env:DASHSCOPE_API_KEY="<REDACTED>"
python -m experiments.react_adapted_v1.run_experiment --mode formal --approve-api --resume
```

`formal_run_status.json` 记录状态为 completed；99 条 `final_records.jsonl` 记录均带相同的 `protocol_sha256=0c2318d80c463a881458c09db9e3e5340c5ab961c23a59d64ad3c42c776555de`，与包内 `experiment_protocol.md` 相符。项目没有保存独立 shell transcript，因此上述命令属于“仓库规定的正式命令”，不能声称为操作系统保留的精确命令历史。

## KDAgent and Serial

SWaT 的精确 shell 命令未找到；历史日志保存了模型、数据路径、RAG、轮数、生成参数、超时和顶层重试配置。WADI 的命令由 `react/implementation/run_experiment.py::main_command` 构造，运行日志证实最终参数，但完整 stdout/shell transcript 未保存。

## Code version

该项目目录不是 Git 仓库，因此没有可验证的实验 commit。`react/results/reproducibility_manifest.json` 保存的是一次文件哈希清单，但其生成时间晚于正式推理完成时间，且不是不可变源码快照。ReAct 每条记录只绑定协议哈希，没有绑定 `react_agent.py` 或 runner 的执行时哈希；因此包内当前代码是现有复现实现，不能严格证明其每一字节都等于执行时版本。服务端也未返回可固定的模型权重 revision/snapshot。
"""
    write_text(EXPORT_ROOT / "run_command_and_version_evidence.md", text)


def write_missing_items() -> None:
    text = """# 缺失项与搜索范围

| Missing or limited item | Searched locations | Safe conclusion |
|---|---|---|
| ReAct 正式运行的独立 shell transcript | `outputs/react_adapted_v1/`, `experiments/react_adapted_v1/`, 项目日志目录 | 未找到；仅保存规定命令、完成状态、逐调用和逐工具记录。 |
| Git commit / tag | 项目根目录及父级 `.git` | 当前项目不是 Git 仓库，不能给出实验 commit。 |
| 执行时完整源码快照 | ReAct manifest、ZIP、final records | final records 只绑定协议哈希；manifest 不是不可变执行时源码快照。 |
| 服务端模型 revision | 模型配置、raw model calls、provider raw responses | 只确认 API model identifier 为 `deepseek-v4-pro`，未找到权重 revision。 |
| SWaT KB 构建 manifest/源指纹 | `outputs/`, `logs/`, `configs/`, `data/rag_kb/` | 未找到与 WADI `kb_build_manifest.json` 等价的文件；只能导出当前 collection、源文件哈希和文件时间。 |
| Markdown chunk 精确字符起止位置 | Chroma metadata、`rag_kb_builder.py` | 未保存；只可给出 chunk 顺序 ID、stage 和源文件。 |
| 知识规则逐段作者与人工核对历史 | 知识文件 front matter、README、构建脚本 | 官方摘要有 URL/DOI；通用规则和模板没有作者编辑/人工核对审计。 |
| 形式化“绝无标签泄漏”证明 | `kb_label_isolation_audit.csv`、测试、raw prompts/tool observations | 现有检查支持结构化字段隔离和可审计轨迹，但不能覆盖所有潜在语义泄漏。 |
| KDAgent/Serial 内部网络尝试的历史独立字段 | 历史日志、case summary、iteration records | 顶层日志为 5；当前 Agent 代码显式覆盖为 3，但历史记录未逐调用保存 `network_attempts`。 |
| KDAgent/Serial 完整大型逐轮原始响应 | 已定位于 frozen_v1/v2 与 WADI comparator `raw_responses/` | 文件存在，但用户清单只明确要求 ReAct 原始日志，本包为控制体积未复制；路径在 README 中保留。 |

没有根据当前默认值补造缺失的历史事实。
"""
    write_text(EXPORT_ROOT / "missing_items_and_search_scope.md", text)


def write_readme(copied: List[Dict[str, Any]], chunks: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> None:
    react_final = EXPORT_ROOT / "react" / "run_records" / "final_records.jsonl"
    with react_final.open("r", encoding="utf-8") as handle:
        final_rows = [json.loads(line) for line in handle if line.strip()]
    counts = Counter(row["dataset"] for row in final_rows)
    stop_counts = Counter(row["stopping_reason"] for row in final_rows)
    source_counts = Counter((row["dataset"], row["source"]) for row in chunks)
    text = f"""# KDAgent Reproducibility Addendum

生成时间：{datetime.now().isoformat()}

本包只读整理现有实验文件。导出过程未调用模型或 Embedding API，未重跑实验，也未修改预测、评分、统计检验或知识库内容。复制文件均通过源文件/副本 SHA-256 一致性检查。

## 完整性摘要

- ReAct：{len(final_rows)} 条正式诊断记录，其中 SWaT {counts['swat']}、WADI {counts['wadi']}；停止状态 {dict(stop_counts)}。
- 实际知识块：{len(chunks)}，其中 SWaT {sum(row['dataset'] == 'swat' for row in chunks)}、WADI {sum(row['dataset'] == 'wadi' for row in chunks)}。
- 知识源文件：{len(sources)} 个 collection-source 组合。
- 结果统计文件保持原始字节，不重新计分。

## 目录说明

- `react/implementation/`：ReAct Agent、工具、提示词、配置、协议、测试和统一入口。
- `react/run_records/`：99 条正式结果、364 次原始模型调用及工具轨迹；`scoring_keys.jsonl` 是在线推理后单独保存的离线标签映射。
- `react/results/`：用户指定的 5 个 CSV，以及正式状态、复现 manifest、数据/隔离审计和重叠敏感性结果。
- `methods/shared_implementation/`：KDAgent、Serial、验证器、评分器、RAG、模型客户端与 WADI adapter 的当前实现。
- `methods/historical_evidence/`：SWaT/WADI KDAgent 与 Serial 的历史运行配置日志及 metrics。
- `knowledge/knowledge_chunks.csv`：从当前 Chroma collections 只读导出的 251 个实际 chunk，含完整内容、ID、来源、位置和 URL/DOI 元数据。
- `knowledge/knowledge_source_catalog.csv`：按数据集和源文件汇总的 chunk 数、源文件哈希与分类。
- `knowledge/*/source_documents/`：构建 collection 使用的变量元数据、工艺模板和补充规则原文。
- `events/swat/`：SWaT 冻结输入、Top-10 摘要和 episode 审计。
- `events/wadi/`：WADI 冻结输入、全部 14 个映射事件（13 纳入、1 排除）、标签、窗口和 Top-10。
- `configuration_comparison.md/.csv`：三种方法配置对照及逐项证据等级。
- `run_command_and_version_evidence.md`：命令与代码版本能证实到的边界。
- `missing_items_and_search_scope.md`：缺失项及实际搜索范围。
- `file_manifest.csv`：包内逐文件来源、用途、大小和 SHA-256。

## 各文件对应的实验

- SWaT KDAgent：`outputs/final_experiments_frozen_v2/dual_branch_fusion_agent_deepseek-v4-pro`，历史 20 episodes x 3 runs。
- SWaT Serial：`outputs/final_experiments_frozen_v1/rag_self_refinement_agent_deepseek-v4-pro`，历史 20 episodes x 3 runs。
- WADI KDAgent：run 1 从 `outputs/wadi_external_validation_v1/kdagent_deepseek-v4-pro` 复用；runs 2-3 写入 `outputs/react_adapted_v1/comparators/wadi/kdagent_deepseek-v4-pro`。
- WADI Serial：`outputs/react_adapted_v1/comparators/wadi/serial_deepseek-v4-pro`，本轮 ReAct 对照阶段生成 3 runs。
- ReAct：`outputs/react_adapted_v1/react`，SWaT 60 + WADI 39，共 99 条。
- `record_level_results.csv` 等联合统计文件由 ReAct v1 的离线分析阶段生成，融合已有 SWaT 对照、WADI 对照和 ReAct 正式记录。

## 知识库来源计数

"""
    source_lines = [f"- {dataset}/{source}: {count} chunks" for (dataset, source), count in sorted(source_counts.items())]
    text += "\n".join(source_lines)
    text += """

## 未打包内容

- Chroma 二进制索引：体积较大且 `knowledge_chunks.csv` 已导出实际内容及 metadata。
- KDAgent/Serial 大型逐轮 raw responses：未在用户必选清单中，且会显著增大包体；其原路径已在 `missing_items_and_search_scope.md` 中说明。
- `.env`、API Key、虚拟环境、`__pycache__`、旧 ZIP、模型 checkpoint 和原始大型传感器数据。

## 使用提醒

`kb_label_isolation_audit.csv` 是现有结构化字段检查记录，不应扩写为对所有语义泄漏的形式化证明。`reproducibility_manifest.json` 也不是 Git commit；具体限制见缺失项报告。
"""
    write_text(EXPORT_ROOT / "README.md", text)


def append_generated_manifest_rows(copied: List[Dict[str, Any]]) -> None:
    for path in sorted(EXPORT_ROOT.rglob("*")):
        if not path.is_file() or path.name == "file_manifest.csv":
            continue
        relative = str(path.relative_to(EXPORT_ROOT)).replace("\\", "/")
        if any(row["package_path"] == relative for row in copied):
            continue
        copied.append(
            {
                "package_path": relative,
                "source_path": "generated_by_export_addendum.py",
                "source_experiment": "reproducibility addendum export; no API and no rescoring",
                "file_role": role_for(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "generated_or_copied": "generated_summary_or_inventory",
            }
        )


def refresh_manifest_row(copied: List[Dict[str, Any]], relative: str) -> None:
    """After a generated file is rewritten at the end, refresh its size and hash in the manifest."""
    path = EXPORT_ROOT / relative
    for row in copied:
        if row["package_path"] == relative:
            row["size_bytes"] = path.stat().st_size
            row["sha256"] = sha256(path)
            return
    raise RuntimeError(f"清单中未找到待刷新文件：{relative}")


def validate_export(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(chunks) != 251:
        raise RuntimeError(f"知识块数异常：{len(chunks)} != 251")
    dataset_chunks = Counter(row["dataset"] for row in chunks)
    if dataset_chunks != Counter({"swat": 121, "wadi": 130}):
        raise RuntimeError(f"知识块数据集分布异常：{dataset_chunks}")

    final_records = EXPORT_ROOT / "react" / "run_records" / "final_records.jsonl"
    with final_records.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    record_counts = Counter(row["dataset"] for row in records)
    if len(records) != 99 or record_counts != Counter({"swat": 60, "wadi": 39}):
        raise RuntimeError(f"ReAct 记录不完整：total={len(records)}, datasets={record_counts}")

    forbidden_suffixes = {".env", ".key", ".pem", ".pyc"}
    forbidden_dirs = {"__pycache__", ".venv", "venv", "chroma_db"}
    secret_patterns = [
        re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
        re.compile(rb"Bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
    ]
    scanned = 0
    for path in EXPORT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in forbidden_suffixes or forbidden_dirs.intersection(path.parts):
            raise RuntimeError(f"包内出现禁止文件：{path}")
        data = path.read_bytes()
        scanned += 1
        if any(pattern.search(data) for pattern in secret_patterns):
            raise RuntimeError(f"检测到疑似密钥模式：{path}")
    return {
        "status": "PASS",
        "api_calls_during_export": 0,
        "react_records": len(records),
        "react_records_by_dataset": dict(record_counts),
        "knowledge_chunks": len(chunks),
        "knowledge_chunks_by_dataset": dict(dataset_chunks),
        "files_secret_scanned": scanned,
        "predictions_or_scores_recomputed": False,
        "knowledge_content_modified": False,
    }


def create_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(EXPORT_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, Path("KDAgent_reproducibility_addendum") / path.relative_to(EXPORT_ROOT))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if EXPORT_ROOT.exists():
        raise RuntimeError(f"导出目录已存在，为避免覆盖请先人工确认处理：{EXPORT_ROOT}")
    EXPORT_ROOT.mkdir(parents=True)

    copied: List[Dict[str, Any]] = []
    for source, target in files_to_copy():
        copy_file(source, target, copied)

    chunks, sources = export_knowledge_chunks()
    export_wadi_episode_table()
    rows = configuration_rows()
    write_configuration_comparison(rows)
    write_run_command_evidence()
    write_missing_items()
    write_readme(copied, chunks, sources)

    validation = validate_export(chunks)
    write_text(EXPORT_ROOT / "export_validation.json", json.dumps(validation, ensure_ascii=False, indent=2))
    append_generated_manifest_rows(copied)
    fields = ["package_path", "source_path", "source_experiment", "file_role", "size_bytes", "sha256", "generated_or_copied"]
    write_csv(EXPORT_ROOT / "file_manifest.csv", sorted(copied, key=lambda row: row["package_path"]), fields)

    # Re-check once more after the manifest is written, to ensure the final package contains no secrets or forbidden files.
    final_validation = validate_export(chunks)
    write_text(EXPORT_ROOT / "export_validation.json", json.dumps(final_validation, ensure_ascii=False, indent=2))
    refresh_manifest_row(copied, "export_validation.json")
    # file_manifest.csv does not record its own hash; every other entry matches the actual bytes written into the package.
    write_csv(EXPORT_ROOT / "file_manifest.csv", sorted(copied, key=lambda row: row["package_path"]), fields)
    validate_export(chunks)
    create_zip()
    print(json.dumps({**final_validation, "export_root": str(EXPORT_ROOT), "zip_path": str(ZIP_PATH), "zip_size_bytes": ZIP_PATH.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
