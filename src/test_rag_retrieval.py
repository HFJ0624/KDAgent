"""Test RAG retrieval on a specified case.

Typical usage:
    python src/test_rag_retrieval.py ^
        --data_path data/swat_s2s_raw_window_fixed/llm_prompt_cases.jsonl ^
        --rag_config configs/rag.yaml ^
        --case_id 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running directly as a script without installing the package
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.rag_retriever import RAGRetriever, build_rag_query  # noqa: E402
from src.utils import load_jsonl, setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在指定案例上测试 RAG 检索。")
    p.add_argument("--data_path", required=True, help="llm_prompt_cases.jsonl 的路径")
    p.add_argument("--rag_config", required=True, help="rag.yaml 的路径")
    p.add_argument("--case_id", type=int, required=True, help="要检索的案例 id")
    p.add_argument("--top_k", type=int, default=None, help="覆盖 rag.yaml 中的 top_k")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(log_dir="logs", name="rag_test")

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        if os.path.exists(p):
            return p
        return os.path.join(_PROJECT_ROOT, p)

    data_path = _resolve(args.data_path)
    rag_config = _resolve(args.rag_config)

    cases = load_jsonl(data_path)
    target_case = None
    for c in cases:
        if int(c.get("case_id", -1)) == args.case_id:
            target_case = c
            break

    if target_case is None:
        print(f"错误：在 {data_path} 中未找到 case_id={args.case_id}", file=sys.stderr)
        sys.exit(2)

    logger.info("已找到案例 %s。", args.case_id)
    logger.info("构造的 RAG 查询：%s", build_rag_query(target_case))

    retriever = RAGRetriever.from_config_path(rag_config, logger=logger)
    result = retriever.retrieve(target_case, top_k=args.top_k)

    print("=" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
