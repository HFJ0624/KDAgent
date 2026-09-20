"""Build the RAG index for SWaT.

Typical usage:
    python src/build_rag_index.py ^
        --data_dir data/swat_s2s_raw_window_fixed ^
        --kb_dir data/rag_kb ^
        --rag_config configs/rag.yaml ^
        --rebuild
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

from src.rag_kb_builder import build_kb  # noqa: E402
from src.utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the SWaT RAG knowledge base.")
    p.add_argument("--data_dir", required=True, help="Directory containing variables_meta.csv and process_knowledge_template.md")
    p.add_argument("--kb_dir", required=True, help="Directory containing supplementary rag_kb/*.md files")
    p.add_argument("--rag_config", required=True, help="Path to rag.yaml")
    p.add_argument("--rebuild", action="store_true", help="Delete the old collection before rebuilding")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(log_dir="logs", name="rag_build")

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        if os.path.exists(p):
            return p
        return os.path.join(_PROJECT_ROOT, p)

    data_dir = _resolve(args.data_dir)
    kb_dir = _resolve(args.kb_dir)
    rag_config = _resolve(args.rag_config)

    result = build_kb(
        rag_config_path=rag_config,
        data_dir=data_dir,
        kb_dir=kb_dir,
        rebuild=args.rebuild,
        logger=logger,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
