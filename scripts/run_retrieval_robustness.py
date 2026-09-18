"""KDAgent Retrieval Robustness experiment.

The default mode only builds the retrieval perturbation plan and does not call the API. Models are
only invoked after ``--run``: KDAgent reuses the clean Evidence branch and reruns only the corrupted
Retrieval branch; Serial RAG+Agent reruns only the corrupted conditions. All outputs are written to
a dedicated directory and support resuming at the ``model/run/case/condition`` granularity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "swat_s2s_raw_window_fixed" / "llm_prompt_cases.jsonl"
KB = ROOT / "data" / "rag_kb"
V2 = ROOT / "outputs" / "final_experiments_frozen_v2"
OUT = ROOT / "outputs" / "retrieval_robustness_v1"
MODELS = ("qwen-plus", "qwen-max", "deepseek-v4-flash", "deepseek-v4-pro", "glm-5.2")
CONDITIONS = ("random_retrieval", "stage_mismatched_retrieval", "low_relevance_retrieval")

sys.path.insert(0, str(ROOT))
from src.data_loader import normalize_case  # noqa: E402
from src.dual_branch_fusion_agent import DualBranchFusionAgent  # noqa: E402
from src.iterative_agent import IterativeSelfRefinementAgent  # noqa: E402
from src.model_client import ModelClient, get_model_config  # noqa: E402
from src.prompt_builder import build_prompt, load_process_knowledge, load_template  # noqa: E402
from src.response_parser import clamp_to_candidates, parse_response  # noqa: E402
from src.utils import read_yaml  # noqa: E402
from src.evaluator import evaluate_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成/运行 Retrieval Robustness 实验")
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--config-path", type=Path, default=ROOT / "configs" / "models.yaml")
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--seed", type=int, default=None, help="随机种子；省略时沿用已有 run_manifest 的 seed")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--run", action="store_true", help="显式调用 API；不加此参数只生成计划")
    p.add_argument("--run-serial", action="store_true", help="在 --run 下同时运行 Serial corrupted 条件")
    p.add_argument("--max-retries", type=int, default=5)
    return p.parse_args()


def load_cases() -> List[Dict[str, Any]]:
    with DATA.open(encoding="utf-8") as f:
        return [normalize_case(json.loads(line)) for line in f if line.strip()]


def tokens(text: str) -> Counter:
    # Preserve both SWaT variable names, English words, and contiguous Chinese fragments so that similarity is
    # not distorted by splitting only on spaces.
    return Counter(re.findall(r"[A-Za-z]+\d+|[A-Za-z]+|[\u4e00-\u9fff]+", text.lower()))


def cosine(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    if not keys: return 0.0
    dot = sum(left[k] * right[k] for k in keys)
    a = math.sqrt(sum(v * v for v in left.values())); b = math.sqrt(sum(v * v for v in right.values()))
    return dot / (a * b) if a and b else 0.0


def stage_of(text: str) -> str:
    upper = text.upper()
    for n in ("1", "2", "3", "4", "5", "6"):
        if re.search(rf"(?:FIT|LIT|AIT|PIT|DPIT|MV|P|UV){n}\d+", upper) or f"P{n}" in upper:
            return n
    return ""


def chunk_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"robust__{Path(source).stem}__{index:05d}__{digest}"


def load_kb() -> List[Dict[str, Any]]:
    """Read document chunks from the existing real KB markdown files; no new knowledge is added or modified."""
    chunks = []
    for source in sorted(KB.glob("*.md")):
        text = source.read_text(encoding="utf-8")
        # Split on heading boundaries to keep each chunk semantically intact; the content still comes entirely from the existing KB.
        parts = [p.strip() for p in re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE) if p.strip()]
        for index, content in enumerate(parts):
            chunks.append({"id": chunk_id(source.name, index, content), "source": source.name, "content": content,
                           "stage": stage_of(content), "tokens": tokens(content)})
    if not chunks: raise RuntimeError(f"KB 为空：{KB}")
    return chunks


def clean_fingerprints() -> set[Tuple[str, str]]:
    """Read the source/content fingerprints of the clean retrieval, used to exclude the clean top-5 chunks."""
    result = set()
    for path in (V2).glob("dual_branch_fusion_agent_*/raw_responses/*_run_*.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("record_type") != "case_summary": continue
                for context in item.get("rag_contexts") or []:
                    result.add((str(context.get("source") or ""), hashlib.sha1(str(context.get("content") or "").encode("utf-8")).hexdigest()))
    return result


def clean_contexts(model: str, run: int, case: int) -> List[Dict[str, Any]]:
    path_root = V2 / f"dual_branch_fusion_agent_{model}" / "raw_responses"
    for path in path_root.glob(f"*_run_{run}.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("record_type") == "case_summary" and int(item.get("case_id")) == case:
                    return list(item.get("rag_contexts") or [])
    return []


def plan_for(case: Mapping[str, Any], kb: List[Dict[str, Any]], clean: set, condition: str, rng: random.Random, top_k: int) -> Dict[str, Any]:
    query = " ".join([str(case.get("task", "")), " ".join(str(x.get("name", "")) for x in case.get("top10_details", [])), "SWaT process root cause analysis"])
    qtokens = tokens(query)
    scored = []
    for item in kb:
        fingerprint = (item["source"], hashlib.sha1(item["content"].encode("utf-8")).hexdigest())
        score = cosine(qtokens, item["tokens"])
        scored.append((score, fingerprint, item))
    candidate_stage = stage_of(" ".join(str(x.get("name", "")) for x in case.get("top10_details", [])))
    available = [x for _, fp, x in scored if fp not in clean]
    fallback = ""
    if condition == "random_retrieval":
        chosen = rng.sample(available, min(top_k, len(available)))
        chosen.sort(key=lambda x: x["id"])
    elif condition == "stage_mismatched_retrieval":
        stage_pool = [x for x in available if candidate_stage and x["stage"] and x["stage"] != candidate_stage]
        if len(stage_pool) < top_k:
            fallback = "insufficient_stage_metadata_or_pool"
            stage_pool = available
        chosen = sorted(stage_pool, key=lambda x: (cosine(qtokens, x["tokens"]), x["id"]))[:top_k]
    else:
        # For the low-relevance condition, exclude documents that clearly contain the current candidate names / the primary
        # stage, then sample from the low-cosine end.
        names = {str(x.get("name", "")).upper() for x in case.get("top10_details", []) if x.get("name")}
        low_pool = [x for score, _, x in scored if x in available and not (names & set(re.findall(r"[A-Z]+\d+", x["content"].upper()))) and (not candidate_stage or x["stage"] != candidate_stage)]
        if len(low_pool) < top_k:
            fallback = "insufficient_exclusion_pool"
            low_pool = available
        chosen = sorted(low_pool, key=lambda x: (cosine(qtokens, x["tokens"]), x["id"]))[:top_k]
    return {"case_id": case["case_id"], "condition": condition, "query": query, "candidate_stage": candidate_stage,
            "fallback": fallback, "contexts": [{"id": x["id"], "content": x["content"], "source": x["source"], "similarity": cosine(qtokens, x["tokens"]), "metadata": {"stage": x["stage"], "source": x["source"], "chunk_id": x["id"]}} for x in chosen]}


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_clean_evidence(model: str, run_id: int, case_id: int) -> Dict[str, Any]:
    """Read the final round of the clean Evidence branch, fully reusing the existing model outputs."""
    root = V2 / f"dual_branch_fusion_agent_{model}" / "raw_responses"
    iterations: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for path in root.glob(f"*_run_{run_id}.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if int(item.get("case_id", -1)) != case_id: continue
                if item.get("record_type") == "case_summary": summary = item
                elif item.get("record_type") == "agent_iteration" and item.get("branch") == "data_evidence": iterations.append(item)
    final_iteration = int(summary.get("final_iteration") or 1)
    selected = [x for x in iterations if int(x.get("iteration") or 1) == final_iteration]
    return selected[-1] if selected else (iterations[-1] if iterations else {})


def append_checkpoint(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def setup_run_logger(output_dir: Path) -> logging.Logger:
    """Write to both the terminal and a dedicated log file, for real-time observation and post-hoc auditing of long experiments."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("retrieval_robustness")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout); console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"retrieval_robustness_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console); logger.addHandler(file_handler)
    return logger


def completed_keys(path: Path) -> set[str]:
    if not path.exists(): return set()
    latest: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                key = str(item.get("record_key"))
                # The same key may be appended as a success after an earlier failure; the last record wins.
                latest[key] = item
            except json.JSONDecodeError: pass
    # Failed, truncated, or empty results are never treated as complete; the next run will retry them automatically.
    def is_success(item: Mapping[str, Any]) -> bool:
        if bool(item.get("was_truncated")): return False
        if item.get("method") == "kdagent":
            # Older fusion records have no api_success, but evidence_reused together with
            # retrieval_record_key marks them as completed deterministic fusion results.
            return bool(item.get("evidence_reused") and item.get("retrieval_record_key"))
        return bool(item.get("api_success"))
    return {key for key, item in latest.items() if is_success(item)}


def latest_record(path: Path, record_key: str) -> Dict[str, Any]:
    """Read the last record for a given checkpoint key, avoiding the use of stale failed results."""
    found: Dict[str, Any] = {}
    if not path.exists(): return found
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("record_key") == record_key: found = item
            except json.JSONDecodeError: continue
    return found


def run_api_experiment(cfg: argparse.Namespace, plans: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], logger: logging.Logger) -> None:
    """Execute corrupted retrieval; each branch/Serial result is written to the checkpoint before continuing."""
    plan_map = {(str(p["model"]), int(p["run_id"]), int(p["case_id"]), str(p["condition"])): p for p in plans}
    case_map = {int(c["case_id"]): dict(c) for c in cases}
    checkpoint = cfg.output_dir / "records.jsonl"; done = completed_keys(checkpoint)
    total = len(cfg.models) * 3 * 20 * len(cfg.conditions); completed = 0
    logger.info("开始/继续实验：目标 retrieval=%d，已有成功 checkpoint=%d，失败记录将自动重试。", total, len(done))
    template = load_template(str(ROOT / "prompts" / "rca_prompt_template.txt"))
    knowledge = load_process_knowledge(str(ROOT / "data" / "process_knowledge_template.md"))
    for model in cfg.models:
        client = ModelClient(get_model_config(str(cfg.config_path), model))
        for run_id in range(1, 4):
            for case_id in range(20):
                case = case_map[case_id]
                evidence = load_clean_evidence(model, run_id, case_id)
                evidence_parsed = evidence.get("parsed_response") or {}
                for condition in cfg.conditions:
                    plan = plan_map[(model, run_id, case_id, condition)]
                    contexts = plan["contexts"]
                    prompt = build_prompt(case, template, process_knowledge=knowledge, rag_contexts=contexts)
                    api_key = f"{model}|{run_id}|{case_id}|{condition}|retrieval_branch"
                    retrieval_changed = False
                    if api_key not in done:
                        old = latest_record(checkpoint, api_key)
                        logger.info("调用 Retrieval branch model=%s condition=%s run=%d case=%d%s", model, condition, run_id, case_id, "（重试失败记录）" if old else "")
                        api = client.chat(prompt=prompt, system_prompt="You are a careful industrial root cause analysis engineer. Output valid JSON only.", max_retries=cfg.max_retries, timeout=120, retry_base_sleep=5)
                        parsed = clamp_to_candidates(parse_response(api.get("content", "")), case.get("top10_vars", []))
                        success = bool(api.get("success")) and api.get("finish_reason", "") != "length"
                        append_checkpoint(checkpoint, {"record_key": api_key, "method": "retrieval_branch", "model": model, "run_id": run_id, "case_id": case_id, "condition": condition, "parsed_response": parsed, "api_success": bool(api.get("success")), "finish_reason": api.get("finish_reason", ""), "was_truncated": api.get("finish_reason", "") == "length", "error_type": api.get("error_type"), "error_message": api.get("error_message"), "prompt_tokens": api.get("prompt_tokens", 0), "completion_tokens": api.get("completion_tokens", 0), "reasoning_tokens": api.get("reasoning_tokens", 0), "total_tokens": api.get("total_tokens", 0), "retrieved_contexts": contexts, "fallback": plan.get("fallback", ""), "plan_seed": plan.get("plan_seed")})
                        retrieval_changed = True
                        if success: done.add(api_key); logger.info("Retrieval 成功 model=%s condition=%s run=%d case=%d", model, condition, run_id, case_id)
                        else: logger.error("Retrieval 失败 model=%s condition=%s run=%d case=%d error=%s: %s", model, condition, run_id, case_id, api.get("error_type"), api.get("error_message"))
                    # KDAgent's Evidence branch is not rerun: the clean branch ranking is combined with the current corrupted
                    # Retrieval branch using two deterministic fusion schemes.
                    latest = latest_record(checkpoint, api_key)
                    retrieval_parsed = latest.get("parsed_response") or {}
                    ev_rank = [str(x).upper() for x in evidence_parsed.get("predicted_root_causes", [])]
                    rt_rank = [str(x).upper() for x in retrieval_parsed.get("predicted_root_causes", [])]
                    top10 = case.get("top10_vars", []); gt = (case.get("ground_truth") or {}).get("gt_vars", [])
                    for fusion in ("authority_preserving", "retrieval_primary"):
                        rank = list(dict.fromkeys((ev_rank[:1] + rt_rank + ev_rank[1:]) if fusion == "authority_preserving" else (rt_rank[:1] + ev_rank + rt_rank[1:])))[:5]
                        primary = rank[0] if rank else ""
                        truth = {str(x).upper() for x in gt}; pos = next((i+1 for i,x in enumerate(rank) if x in truth), None)
                        key = f"{model}|{run_id}|{case_id}|{condition}|kdagent_{fusion}"
                        if retrieval_changed or key not in done:
                            append_checkpoint(checkpoint, {"record_key": key, "method": "kdagent", "fusion": fusion, "model": model, "run_id": run_id, "case_id": case_id, "condition": condition, "predicted_root_causes": rank, "primary_root_cause": primary, "hit_at_1": int(pos == 1), "hit_at_3": int(bool(pos and pos <= 3)), "hit_at_5": int(bool(pos and pos <= 5)), "mrr": 1/pos if pos else 0.0, "best_gt_rank": pos or "", "retrieved_contexts": contexts, "fallback": plan.get("fallback", ""), "evidence_reused": True, "retrieval_record_key": api_key})
                            done.add(key)
                    if cfg.run_serial:
                        key = f"{model}|{run_id}|{case_id}|{condition}|serial"
                        if retrieval_changed or key not in done:
                            serial = IterativeSelfRefinementAgent(model_client=client, min_confidence=0.5)
                            result = serial.run_case(case=case, initial_prompt=prompt, top10_vars=top10, rag_contexts=contexts, max_iterations=3, run_id=run_id)
                            parsed = result.get("parsed_response", {}) or {}; rank = parsed.get("predicted_root_causes", []) or []
                            truth = {str(x).upper() for x in gt}; pos = next((i+1 for i,x in enumerate(rank) if str(x).upper() in truth), None)
                            append_checkpoint(checkpoint, {"record_key": key, "method": "serial", "model": model, "run_id": run_id, "case_id": case_id, "condition": condition, "predicted_root_causes": rank, "primary_root_cause": parsed.get("primary_root_cause", ""), "hit_at_1": int(pos == 1), "hit_at_3": int(bool(pos and pos <= 3)), "hit_at_5": int(bool(pos and pos <= 5)), "mrr": 1/pos if pos else 0.0, "best_gt_rank": pos or "", "iteration_count": result.get("iteration_count", 1), "retrieved_contexts": contexts, "fallback": plan.get("fallback", "")})
                            done.add(key)
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        logger.info("进度：%d/%d 个 retrieval 组合，checkpoint=%s", completed, total, checkpoint)
    print(f"已完成 corrupted retrieval 实验，checkpoint：{checkpoint}")


def make_plans(cases, kb, clean, cfg) -> List[Dict[str, Any]]:
    rows = []
    for model in cfg.models:
        for run in range(1, 4):
            for case in cases:
                for condition in cfg.conditions:
                    rng = random.Random(cfg.seed + run * 10000 + int(case["case_id"]) * 100 + list(CONDITIONS).index(condition))
                    row = plan_for(case, kb, clean, condition, rng, cfg.top_k); row.update({"model": model, "run_id": run, "plan_seed": cfg.seed + run * 10000 + int(case["case_id"]) * 100 + list(CONDITIONS).index(condition)})
                    rows.append(row)
    return rows


def main() -> None:
    cfg = parse_args(); cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(cfg.output_dir)
    cases, kb, clean = load_cases(), load_kb(), clean_fingerprints()
    if cfg.seed is None:
        manifest = cfg.output_dir / "run_manifest.json"
        if manifest.exists():
            try: cfg.seed = int(json.loads(manifest.read_text(encoding="utf-8")).get("seed", 20260827))
            except (ValueError, TypeError, json.JSONDecodeError): cfg.seed = 20260827
        else: cfg.seed = 20260827
    plans = make_plans(cases, kb, clean, cfg)
    write_jsonl(cfg.output_dir / "retrieval_plans.jsonl", plans)
    (cfg.output_dir / "run_manifest.json").write_text(json.dumps({"benchmark_cases": 20, "models": cfg.models, "runs": 3, "conditions": cfg.conditions, "seed": cfg.seed, "kb_sources": sorted({x["source"] for x in kb}), "clean_exclusion_fingerprints": len(clean), "run": cfg.run, "run_serial": cfg.run_serial, "api_calls": 0 if not cfg.run else "pending"}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cfg.run:
        logger.info("仅生成检索计划（未调用 API）：%s", cfg.output_dir)
        return
    run_api_experiment(cfg, plans, cases, logger)


if __name__ == "__main__": main()
