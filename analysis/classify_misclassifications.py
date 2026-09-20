"""Bucket misclassifications in an RCA run into interpretable error types.

Reads the per-case parsed-results CSV produced by ``src/main.py`` (one row per
case x run) and, for every row that is not a top-1 hit, assigns it to a slice:

- ``rank_error_top5``   : the ground-truth variable is in the top 5 predictions
                          but is not ranked first (ordering failure).
- ``beyond_top5``       : the ground-truth variable is predicted but ranked
                          beyond position 5 (coverage failure).
- ``omission``          : the ground-truth variable is absent from the
                          predictions entirely (recall failure).
- ``hallucination``     : the model predicted a variable outside the candidate
                          Top-10 search space (constraint violation).
- ``invalid_parse``     : the response parsed but carried no usable primary.
- ``api_failure``       : the API call or error path produced no prediction.

Only the final evaluated row of an evaluated case must be treated as a "hit";
all rows are bucketed against the same ground-truth normalization used by the
evaluator (uppercase IDs, comma-separated).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# Helpers (import-free so the script works from any directory when data is ready)
# -----------------------------------------------------------------------------


def _norm_csv(value) -> List[str]:
    """Split a comma-separated CSV cell into normalized uppercase IDs."""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def _norm_set(value) -> set:
    return {item for item in _norm_csv(value)}


_BUCKETS = ["hit_at_1", "rank_error_top5", "beyond_top5",
            "omission", "hallucination", "invalid_parse", "api_failure"]


def classify_row(row: Dict[str, object]) -> str:
    """Return the error-type bucket for a single parsed-results row."""
    api_ok = str(row.get("api_success", "1"))  # 0 -> failed, 1 -> ok
    error_type = str(row.get("error_type", "") or "")
    if api_ok == "0" or error_type:
        return "api_failure"

    predicted = _norm_set(row.get("predicted_root_causes"))
    gt = _norm_set(row.get("gt_vars"))
    top10 = _norm_set(row.get("top10_vars"))
    is_hit1 = int(row.get("is_hit_at_1", 0) or 0)
    is_hit5 = int(row.get("is_hit_at_5", 0) or 0)

    if is_hit1:
        return "hit_at_1"
    # Hallucination: predicting something that is not even a candidate.
    if predicted and not predicted.issubset(top10):
        return "hallucination"
    # Could not parse a usable primary -> no prediction at all.
    primary = str(row.get("primary_root_cause", "") or "").strip().upper()
    if not predicted and not primary:
        return "invalid_parse"
    if is_hit5:
        return "rank_error_top5"     # gt present in top-5, wrong order
    if gt & predicted:
        return "beyond_top5"         # gt present but ranked > 5
    return "omission"                # gt absent from predictions


# -----------------------------------------------------------------------------
# Aggregation and reporting
# -----------------------------------------------------------------------------


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Count buckets globally and per model."""
    global_counts: Counter = Counter()
    model_counts: Dict[str, Counter] = defaultdict(Counter)
    total = 0
    for row in rows:
        total += 1
        bucket = classify_row(row)
        global_counts[bucket] += 1
        model = str(row.get("model_name", "") or "(unknown)")
        model_counts[model][bucket] += 1

    def _safe_share(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    return {
        "total_rows": total,
        "global_counts": dict(global_counts),
        "global_shares": {k: _safe_share(global_counts[k]) for k in _BUCKETS},
        "per_model": {
            model: {
                "counts": dict(counts),
                "shares": {k: round(counts[k] / (sum(counts.values()) or 1), 4) for k in _BUCKETS},
            }
            for model, counts in model_counts.items()
        },
    }


def format_report(report: Dict[str, object]) -> str:
    lines: List[str] = []
    g = report["global_counts"]
    lines.append("=== Misclassification breakdown (all rows) ===")
    for bucket in _BUCKETS:
        lines.append(f"  {bucket:<16} {g.get(bucket, 0):>6}")
    lines.append(f"{'total':<16} {report['total_rows']:>6}")
    for model, payload in report["per_model"].items():
        lines.append(f"\n--- {model} ---")
        for bucket in _BUCKETS:
            lines.append(f"  {bucket:<16} {payload['counts'].get(bucket, 0):>6}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify RCA misclassifications into interpretable error types."
    )
    parser.add_argument("--parsed-csv", required=True,
                        help="Path to parsed_results/<file>.csv produced by src/main.py.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write summary.json/summary.csv (default: same dir as the input).")
    args = parser.parse_args()

    try:
        import pandas as pd  # type: ignore
    except ImportError:
        raise SystemExit("pandas is required to read the parsed-results CSV.")

    df = pd.read_csv(args.parsed_csv, dtype=str, keep_default_na=False)
    rows: List[Dict[str, object]] = df.to_dict(orient="records")
    report = aggregate(rows)

    print(format_report(report))

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.parsed_csv))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "misclassification_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Flatten per-model counts into a CSV too.
    flat = []
    for model, payload in report["per_model"].items():
        base = {"model_name": model}
        for bucket in _BUCKETS:
            base[f"count_{bucket}"] = payload["counts"].get(bucket, 0)
            base[f"share_{bucket}"] = payload["shares"].get(bucket, 0)
        flat.append(base)
    pd.DataFrame(flat, columns=None).to_csv(
        os.path.join(out_dir, "misclassification_summary.csv"), index=False,
        encoding="utf-8-sig",
    )
    print(f"\nWrote misclassification_summary.json and .csv to: {out_dir}")


if __name__ == "__main__":
    main()