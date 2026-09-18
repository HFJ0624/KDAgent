"""Recover the auditable timing information for the enhanced AERCA episodes from the raw SWaT Attack workbook."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


@dataclass
class AttackInterval:
    """A contiguous Attack-labelled interval in the raw workbook."""

    interval_no: int
    start_data_index: int
    end_data_index: int
    start_excel_row: int
    end_excel_row: int
    start_timestamp: object
    end_timestamp: object

    @property
    def sample_count(self) -> int:
        return self.end_data_index - self.start_data_index + 1


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "")


def normalize_header(value: object) -> str:
    return str(value).strip() if value is not None else ""


def load_episode_series(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    summary = pd.read_csv(data_dir / "case_top10_summary.csv")
    series = pd.read_csv(
        data_dir / "case_s2s_table.csv",
        usecols=["case_id", "rank", "original_name", "local_time_index", "raw_value"],
    )

    case_series: dict[int, dict[str, np.ndarray]] = {}
    for case_id, case_rows in series.groupby("case_id", sort=True):
        values_by_name: dict[str, np.ndarray] = {}
        for name, variable_rows in case_rows.groupby("original_name", sort=False):
            ordered = variable_rows.sort_values("local_time_index")
            values_by_name[str(name)] = ordered["raw_value"].to_numpy(dtype=np.float64)
        case_series[int(case_id)] = values_by_name
    return summary, series, case_series


def scan_raw_workbook(
    workbook_path: Path,
    required_variables: list[str],
) -> tuple[list[str], list[object], np.ndarray, list[str], list[AttackInterval], dict[str, int]]:
    """Stream-read the large xlsx, keeping only variables matching those required, to avoid modifying or copying the raw workbook."""

    wb = load_workbook(workbook_path, read_only=True, data_only=True)
    ws = wb["Combined Data"]
    rows = ws.iter_rows(values_only=True)

    first_row = next(rows)
    second_row = next(rows)
    headers = [normalize_header(v) for v in (second_row if not any(first_row) else first_row)]
    header_map = {name: index for index, name in enumerate(headers) if name}

    missing = sorted(set(required_variables) - set(header_map))
    if missing:
        raise ValueError(f"原始 SWaT 工作簿缺少变量列: {missing}")
    timestamp_col = header_map.get("Timestamp")
    label_col = header_map.get("Normal/Attack")
    if timestamp_col is None or label_col is None:
        raise ValueError("未找到 Timestamp 或 Normal/Attack 列")

    selected_cols = [header_map[name] for name in required_variables]
    timestamps: list[object] = []
    labels: list[str] = []
    value_rows: list[list[float]] = []
    intervals: list[AttackInterval] = []
    active_start: tuple[int, int, object] | None = None

    for data_index, row in enumerate(rows):
        timestamp = row[timestamp_col]
        label = normalize_label(row[label_col])
        timestamps.append(timestamp)
        labels.append(label)
        value_rows.append([
            float(row[col]) if row[col] is not None and row[col] != "" else math.nan
            for col in selected_cols
        ])

        is_attack = label == "attack"
        excel_row = data_index + 3
        if is_attack and active_start is None:
            active_start = (data_index, excel_row, timestamp)
        elif not is_attack and active_start is not None:
            start_index, start_row, start_time = active_start
            intervals.append(
                AttackInterval(
                    interval_no=len(intervals) + 1,
                    start_data_index=start_index,
                    end_data_index=data_index - 1,
                    start_excel_row=start_row,
                    end_excel_row=excel_row - 1,
                    start_timestamp=start_time,
                    end_timestamp=timestamps[-2],
                )
            )
            active_start = None

        if data_index and data_index % 50000 == 0:
            print(f"[scan] 已读取 {data_index:,} 条 SWaT 数据", flush=True)

    if active_start is not None:
        start_index, start_row, start_time = active_start
        intervals.append(
            AttackInterval(
                interval_no=len(intervals) + 1,
                start_data_index=start_index,
                end_data_index=len(timestamps) - 1,
                start_excel_row=start_row,
                end_excel_row=len(timestamps) + 2,
                start_timestamp=start_time,
                end_timestamp=timestamps[-1],
            )
        )

    wb.close()
    values = np.asarray(value_rows, dtype=np.float64)
    variable_to_matrix_col = {name: i for i, name in enumerate(required_variables)}
    return headers, timestamps, values, labels, intervals, variable_to_matrix_col


def choose_signature_variables(values_by_name: dict[str, np.ndarray], count: int = 3) -> list[str]:
    """Prefer continuous variables rich in variation, reducing accidental matches caused by stable switch-like values."""

    scored = []
    for name, values in values_by_name.items():
        finite = values[np.isfinite(values)]
        unique_count = len(np.unique(finite))
        spread = float(np.nanstd(finite)) if finite.size else 0.0
        scored.append((unique_count, spread, name))
    scored.sort(reverse=True)
    return [name for _, _, name in scored[:count]]


def candidate_starts_for_case(
    raw_values: np.ndarray,
    variable_to_col: dict[str, int],
    values_by_name: dict[str, np.ndarray],
    raw_stride: int,
) -> np.ndarray:
    length = len(next(iter(values_by_name.values())))
    raw_span = (length - 1) * raw_stride + 1
    max_start = raw_values.shape[0] - raw_span
    if max_start < 0:
        return np.array([], dtype=np.int64)

    starts = np.arange(max_start + 1, dtype=np.int64)
    signature_names = choose_signature_variables(values_by_name)
    offsets = sorted(set([0, length // 4, length // 2, (3 * length) // 4, length - 1]))
    keep = np.ones(starts.shape[0], dtype=bool)

    for name in signature_names:
        expected = values_by_name[name]
        raw_col = raw_values[:, variable_to_col[name]]
        for offset in offsets:
            target = expected[offset]
            if not np.isfinite(target):
                continue
            keep &= np.isclose(
                raw_col[starts + offset * raw_stride],
                target,
                rtol=1e-7,
                atol=1e-7,
                equal_nan=False,
            )
            if not keep.any():
                return np.array([], dtype=np.int64)
        starts = starts[keep]
        keep = np.ones(starts.shape[0], dtype=bool)
    return starts


def score_candidate(
    start: int,
    raw_values: np.ndarray,
    variable_to_col: dict[str, int],
    values_by_name: dict[str, np.ndarray],
    raw_stride: int,
) -> tuple[float, float]:
    """Re-verify matching over the full window using all Top-10 candidate variables, rather than relying on only a few signature points."""

    absolute_errors: list[np.ndarray] = []
    exact_flags: list[np.ndarray] = []
    for name, expected in values_by_name.items():
        stop = start + len(expected) * raw_stride
        actual = raw_values[start:stop:raw_stride, variable_to_col[name]]
        finite = np.isfinite(expected) & np.isfinite(actual)
        if finite.any():
            absolute_errors.append(np.abs(actual[finite] - expected[finite]))
            exact_flags.append(np.isclose(actual[finite], expected[finite], rtol=1e-7, atol=1e-7))
    if not absolute_errors:
        return math.inf, 0.0
    errors = np.concatenate(absolute_errors)
    flags = np.concatenate(exact_flags)
    return float(np.sqrt(np.mean(errors**2))), float(np.mean(flags))


def find_interval(intervals: list[AttackInterval], center_index: int) -> AttackInterval | None:
    for interval in intervals:
        if interval.start_data_index <= center_index <= interval.end_data_index:
            return interval
    return None


def format_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return str(value) if value is not None else ""


def normalize_variable_name(value: object) -> str:
    return str(value).strip().replace("-", "").replace(" ", "").upper()


def load_official_attack_list(path: Path) -> pd.DataFrame:
    attacks = pd.read_excel(path)
    attacks = attacks.dropna(subset=["Attack #", "Start Time", "End Time"]).copy()
    attacks["Attack #"] = attacks["Attack #"].astype(int)
    return attacks.set_index("Attack #", drop=False)


def build_official_attack_audit(
    official_attacks: pd.DataFrame,
    timestamps: list[object],
    labels: list[str],
) -> tuple[list[dict[str, object]], pd.Series]:
    raw_timestamps = pd.to_datetime(
        pd.Series(timestamps, dtype="string").str.strip(),
        format="%d/%m/%Y %I:%M:%S %p",
    )
    attack_mask = np.asarray(labels) == "attack"
    rows: list[dict[str, object]] = []
    for attack_id, attack in official_attacks.iterrows():
        start = pd.Timestamp(attack["Start Time"])
        end_time = attack["End Time"]
        end = pd.Timestamp.combine(start.date(), end_time)
        matched_attack_rows = int(((raw_timestamps >= start) & (raw_timestamps <= end) & attack_mask).sum())
        included = int(attack_id) <= 19 and matched_attack_rows > 0
        if included:
            status = "Included"
            reason = "加载器按 attack-list 顺序生成样本；固定导出中对应 source_sample_index 0-19。"
        elif matched_attack_rows == 0:
            status = "Excluded"
            reason = "按源表时间与原始数据匹配时 Attack 行数为 0；加载器据此跳过该事件。"
        else:
            status = "Not exported"
            reason = "存在可匹配原始 Attack 行，但不在当前固定 20-case 导出中。"
        rows.append(
            {
                "official_attack_id": int(attack_id),
                "source_start_timestamp": format_timestamp(start.to_pydatetime()),
                "source_end_time": format_timestamp(end_time),
                "combined_end_timestamp": format_timestamp(end.to_pydatetime()),
                "official_attacked_variables": str(attack["Attack Point"]),
                "matched_raw_attack_rows": matched_attack_rows,
                "benchmark_status": status,
                "status_reason": reason,
                "date_outside_raw_range": "Yes" if start < raw_timestamps.min() or start > raw_timestamps.max() else "No",
            }
        )
    return rows, raw_timestamps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-workbook", type=Path, required=True)
    parser.add_argument("--episode-data-dir", type=Path, required=True)
    parser.add_argument("--attack-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-stride", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, _, case_series = load_episode_series(args.episode_data_dir)
    official_attacks = load_official_attack_list(args.attack_list)
    required_variables = sorted({name for values in case_series.values() for name in values})
    print(f"[prepare] 20 个 episode 共需匹配 {len(required_variables)} 个原始变量", flush=True)

    headers, timestamps, raw_values, labels, intervals, variable_to_col = scan_raw_workbook(
        args.raw_workbook, required_variables
    )
    official_attack_rows, parsed_raw_timestamps = build_official_attack_audit(
        official_attacks, timestamps, labels
    )
    print(
        f"[scan] 完成：{len(timestamps):,} 条记录，{len(intervals)} 个连续 Attack 区间",
        flush=True,
    )

    audit_rows: list[dict[str, object]] = []
    match_details: list[dict[str, object]] = []
    for row in summary.to_dict(orient="records"):
        case_id = int(row["case_id"])
        values_by_name = case_series[case_id]
        length = int(row["context_length"])
        candidates = candidate_starts_for_case(
            raw_values, variable_to_col, values_by_name, args.raw_stride
        )
        scored = [
            (
                int(start),
                *score_candidate(
                    int(start), raw_values, variable_to_col, values_by_name, args.raw_stride
                ),
            )
            for start in candidates
        ]
        scored.sort(key=lambda item: (item[1], -item[2], item[0]))

        # The acceptance condition matches the point-wise np.isclose check exactly, avoiding an additional absolute RMSE threshold with a mismatched scale.
        accepted = [item for item in scored if item[2] >= 0.999999]
        unique = len(accepted) == 1
        best = accepted[0] if accepted else (scored[0] if scored else None)
        if best is not None:
            start_index, rmse, exact_rate = best
            center_index = start_index + int(row["center_time_index"]) * args.raw_stride
            end_index = start_index + (length - 1) * args.raw_stride
            interval = find_interval(intervals, center_index)
            match_status = "唯一精确匹配" if unique else ("多个精确匹配" if accepted else "未达到精确匹配阈值")
        else:
            start_index = center_index = end_index = -1
            rmse = math.nan
            exact_rate = math.nan
            interval = None
            match_status = "无候选匹配"

        gt_names = str(row["gt_names"])
        gt_count = len([value for value in gt_names.split(";") if value])
        best_gt_rank = int(row["best_gt_rank"])
        source_sample_index = int(row["source_sample_index"])
        official = official_attacks.loc[source_sample_index]
        official_points = [
            normalize_variable_name(value) for value in str(official["Attack Point"]).split(",")
        ]
        gt_normalized = {normalize_variable_name(value) for value in gt_names.split(";")}
        audit_rows.append(
            {
                "episode_id": case_id,
                "source_sample_index": source_sample_index,
                "official_attack_id": int(official["Attack #"]),
                "official_attack_start": format_timestamp(official["Start Time"]),
                "official_attack_end_time": format_timestamp(official["End Time"]),
                "official_attacked_variables": str(official["Attack Point"]),
                "official_start_state": str(official["Start State"]) if pd.notna(official["Start State"]) else "",
                "official_attack_action": str(official["Attack"]) if pd.notna(official["Attack"]) else "",
                "official_actual_change": str(official["Actual Change"]) if pd.notna(official["Actual Change"]) else "",
                "official_expected_impact": str(official["Expected Impact or attacker intent"]) if pd.notna(official["Expected Impact or attacker intent"]) else "",
                "official_unexpected_outcome": str(official["Unexpected Outcome"]) if pd.notna(official["Unexpected Outcome"]) else "",
                "official_attack_points_in_gt": "Yes" if set(official_points).issubset(gt_normalized) else "No",
                "center_binary_attack_interval_no": interval.interval_no if interval else "",
                "center_binary_interval_start": format_timestamp(interval.start_timestamp) if interval else "",
                "center_binary_interval_end": format_timestamp(interval.end_timestamp) if interval else "",
                "episode_window_start": format_timestamp(timestamps[start_index]) if start_index >= 0 else "",
                "episode_center_timestamp": format_timestamp(timestamps[center_index]) if center_index >= 0 else "",
                "episode_window_end": format_timestamp(timestamps[end_index]) if end_index >= 0 else "",
                "raw_start_data_index_0based": start_index if start_index >= 0 else "",
                "raw_center_data_index_0based": center_index if center_index >= 0 else "",
                "raw_end_data_index_0based": end_index if end_index >= 0 else "",
                "raw_start_excel_row": start_index + 3 if start_index >= 0 else "",
                "raw_center_excel_row": center_index + 3 if center_index >= 0 else "",
                "raw_end_excel_row": end_index + 3 if end_index >= 0 else "",
                "window_length": length,
                "raw_sampling_stride_seconds": args.raw_stride,
                "ground_truth_var_ids": str(row["gt_vars"]),
                "ground_truth_names": gt_names,
                "label_type": "multi-label" if gt_count > 1 else "single-label",
                "best_gt_rank_in_ta_rca": best_gt_rank,
                "ta_rca_top10_covered": "Yes" if best_gt_rank <= 10 else "No",
                "frozen_ta_rca_top10": str(row["top10_names"]),
                "match_status": match_status,
                "exact_match_count": len(accepted),
                "match_rmse": rmse,
                "match_exact_rate": exact_rate,
                "center_raw_label": labels[center_index] if center_index >= 0 else "",
            }
        )
        match_details.append(
            {
                "episode_id": case_id,
                "signature_variables": choose_signature_variables(values_by_name),
                "candidate_count_after_signature": len(candidates),
                "exact_match_count": len(accepted),
                "top_candidates": [
                    {"start_data_index": s, "rmse": e, "exact_rate": r} for s, e, r in scored[:10]
                ],
            }
        )
        print(f"[match] episode {case_id:02d}: {match_status}，精确候选={len(accepted)}", flush=True)

    interval_rows = [
        {
            "interval_no": interval.interval_no,
            "start_timestamp": format_timestamp(interval.start_timestamp),
            "end_timestamp": format_timestamp(interval.end_timestamp),
            "start_data_index_0based": interval.start_data_index,
            "end_data_index_0based": interval.end_data_index,
            "start_excel_row": interval.start_excel_row,
            "end_excel_row": interval.end_excel_row,
            "sample_count": interval.sample_count,
        }
        for interval in intervals
    ]

    pd.DataFrame(audit_rows).to_csv(args.output_dir / "swat_episode_audit.csv", index=False, encoding="utf-8")
    pd.DataFrame(interval_rows).to_csv(args.output_dir / "swat_attack_intervals.csv", index=False, encoding="utf-8")
    pd.DataFrame(official_attack_rows).to_csv(
        args.output_dir / "swat_official_attack_audit.csv", index=False, encoding="utf-8"
    )
    (args.output_dir / "match_details.json").write_text(
        json.dumps(match_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "raw_workbook": str(args.raw_workbook),
        "official_attack_list": str(args.attack_list),
        "episode_data_dir": str(args.episode_data_dir),
        "raw_row_count": len(timestamps),
        "raw_timestamp_min": format_timestamp(parsed_raw_timestamps.min().to_pydatetime()),
        "raw_timestamp_max": format_timestamp(parsed_raw_timestamps.max().to_pydatetime()),
        "raw_header_count": len(headers),
        "attack_interval_count": len(intervals),
        "episode_count": len(audit_rows),
        "raw_sampling_stride_seconds": args.raw_stride,
        "unique_exact_match_count": sum(row["match_status"] == "唯一精确匹配" for row in audit_rows),
        "official_attack_row_count": len(official_attack_rows),
        "official_attack_rows_with_raw_match": sum(
            row["matched_raw_attack_rows"] > 0 for row in official_attack_rows
        ),
        "note": "attack interval 是二值标签的连续区间；不是官方 attack ID。",
    }
    (args.output_dir / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
