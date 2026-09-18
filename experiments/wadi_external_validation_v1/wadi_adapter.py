"""Minimal adapter layer from raw WADI files to frozen episode panels."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


META_COLUMNS = {"row", "date", "time"}


@dataclass(frozen=True)
class WadiFiles:
    """Absolute paths of WADI's three input files."""

    normal: Path
    attack: Path
    mapping: Path


def locate_wadi_files(data_dir: Path) -> WadiFiles:
    """Locate the input files and validate them in one pass before entering the time-consuming stage."""
    files = WadiFiles(
        normal=data_dir / "WADI_14days_new.csv",
        attack=data_dir / "WADI_attackdataLABLE.csv",
        mapping=data_dir / "wadi_attack_mapping.csv",
    )
    missing = [str(path) for path in files.__dict__.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"WADI 输入文件缺失：{missing}")
    return files


def _clean_column(value: Any) -> str:
    return str(value).strip()


def _normalized_tag(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", _clean_column(value).upper())


def _read_mapping(path: Path) -> pd.DataFrame:
    """The mapping table comes from an Excel export; read it by trying a fixed candidate order of encodings."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030", "gbk", "latin1"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            frame.columns = [_clean_column(column) for column in frame.columns]
            required = {
                "attack_id",
                "start_time",
                "end_time",
                "root_tags",
                "affected_tags",
                "notes",
            }
            if not required.issubset(frame.columns):
                raise ValueError(f"映射表缺少字段：{sorted(required - set(frame.columns))}")
            return frame
        except (UnicodeDecodeError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"无法读取 WADI attack mapping：{last_error}")


def _read_headers(files: WadiFiles) -> Tuple[List[str], Dict[str, str], Dict[str, str], str]:
    """Return the shared feature names, the normal/attack raw-column name mappings, and the attack label column."""
    normal_raw = list(pd.read_csv(files.normal, nrows=0).columns)
    # The attack file's first row is 0..130; the second row is the real header.
    attack_raw = list(pd.read_csv(files.attack, header=1, nrows=0).columns)
    normal_by_clean = {_clean_column(column): str(column) for column in normal_raw}
    attack_by_clean = {_clean_column(column): str(column) for column in attack_raw}

    label_candidates = [
        column
        for column in attack_by_clean
        if "attack" in column.lower()
        and ("label" in column.lower() or "lable" in column.lower())
    ]
    if len(label_candidates) != 1:
        raise ValueError(f"攻击标签列数量异常：{label_candidates}")
    label_column = label_candidates[0]

    features = [
        column
        for column in normal_by_clean
        if column.lower() not in META_COLUMNS
        and column in attack_by_clean
        and column != label_column
    ]
    if not features:
        raise ValueError("正常文件与攻击文件没有公共过程变量。")
    if len(features) != len(set(features)):
        raise ValueError("清理空白后出现重复 WADI 变量名。")
    return features, normal_by_clean, attack_by_clean, label_column


def build_variable_catalog(files: WadiFiles) -> List[Dict[str, Any]]:
    """Build the W000..W126 bidirectional mapping in the stable column order of the normal file."""
    features, _, _, _ = _read_headers(files)
    catalog: List[Dict[str, Any]] = []
    for index, name in enumerate(features):
        catalog.append(
            {
                "var_id": f"W{index:03d}",
                "var_index": index,
                "original_name": name,
                "stage": infer_stage(name),
                "role": infer_role(name),
                "description": describe_variable(name),
            }
        )
    return catalog


def _parse_attack_flags(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        raise ValueError("Attack Label 无法解析为数值。")
    return (numeric.to_numpy() == -1).astype(np.int8)


def contiguous_intervals(flags: Sequence[int]) -> List[Tuple[int, int]]:
    """Convert the binary labels into non-overlapping closed intervals without window augmentation."""
    values = np.asarray(flags, dtype=np.int8)
    padded = np.pad(values, (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _parse_tag_list(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = _clean_column(value)
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in re.split(r"[;；,\n]+", text) if item.strip()]


def _display_attack_id(value: Any) -> str:
    """Keep the original information while normalizing garbled separators into a readable combination of numbers."""
    numbers = re.findall(r"\d+", _clean_column(value))
    return "&".join(numbers) if numbers else _clean_column(value)


def _repair_declared_datetime(value: Any) -> str:
    """Fix only the known Excel date escaping in the mapping table; it is not used for episode alignment."""
    text = _clean_column(value)
    damaged = re.match(r"^(2009|2010|2011)/10/17\s+(.+)$", text)
    if damaged:
        repaired = f"2017-10-{int(damaged.group(1)) - 2000:02d} {damaged.group(2)}"
        parsed = pd.to_datetime(repaired, errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return ""
    return parsed.isoformat(sep=" ")


def build_episode_manifest(files: WadiFiles) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Pair contiguous attack intervals one-to-one with the mapping's fixed row order, producing an audit manifest."""
    catalog = build_variable_catalog(files)
    _, _, attack_by_clean, label_column = _read_headers(files)
    raw_meta_columns = [
        attack_by_clean[column]
        for column in attack_by_clean
        if column.lower() in META_COLUMNS
    ]
    raw_label = attack_by_clean[label_column]
    attack_meta = pd.read_csv(
        files.attack,
        header=1,
        usecols=raw_meta_columns + [raw_label],
        dtype=str,
        low_memory=False,
    )
    attack_meta.columns = [_clean_column(column) for column in attack_meta.columns]
    intervals = contiguous_intervals(_parse_attack_flags(attack_meta[label_column]))
    mapping = _read_mapping(files.mapping)
    if len(intervals) != len(mapping):
        raise ValueError(
            "连续攻击区间与 mapping 行数不一致，禁止按顺序猜测配对："
            f"{len(intervals)} != {len(mapping)}"
        )

    alias_to_id = {
        _normalized_tag(item["original_name"]): item["var_id"] for item in catalog
    }
    rows: List[Dict[str, Any]] = []
    for index, ((start, end), (_, mapping_row)) in enumerate(
        zip(intervals, mapping.iterrows()), start=1
    ):
        declared_roots = _parse_tag_list(mapping_row["root_tags"])
        resolved = [
            alias_to_id[_normalized_tag(tag)]
            for tag in declared_roots
            if _normalized_tag(tag) in alias_to_id
        ]
        unresolved = [
            tag for tag in declared_roots if _normalized_tag(tag) not in alias_to_id
        ]
        included = bool(declared_roots) and not unresolved
        if not declared_roots:
            exclusion_reason = "missing_root_tags"
        elif unresolved:
            exclusion_reason = "unresolved_root_tags"
        else:
            exclusion_reason = ""

        start_meta = attack_meta.iloc[start]
        end_meta = attack_meta.iloc[end]
        rows.append(
            {
                "episode_id": f"WADI-E{index:02d}",
                "case_id": f"WADI-E{index:02d}",
                "mapping_order": index,
                "source_attack_id": _display_attack_id(mapping_row["attack_id"]),
                "attack_row_start": start,
                "attack_row_end": end,
                "attack_row_count": end - start + 1,
                "source_row_start": _clean_column(start_meta.get("Row", "")),
                "source_row_end": _clean_column(end_meta.get("Row", "")),
                "raw_date_start": _clean_column(start_meta.get("Date", "")),
                "raw_time_start": _clean_column(start_meta.get("Time", "")),
                "raw_date_end": _clean_column(end_meta.get("Date", "")),
                "raw_time_end": _clean_column(end_meta.get("Time", "")),
                "declared_start_time": _repair_declared_datetime(mapping_row["start_time"]),
                "declared_end_time": _repair_declared_datetime(mapping_row["end_time"]),
                "root_tags": ";".join(declared_roots),
                "gt_vars": ";".join(dict.fromkeys(resolved)),
                "gt_names": ";".join(declared_roots) if included else "",
                "label_cardinality": len(set(resolved)),
                "label_type": (
                    "unlabeled"
                    if not resolved
                    else "multi-label"
                    if len(set(resolved)) > 1
                    else "single-label"
                ),
                "affected_tags": ";".join(_parse_tag_list(mapping_row["affected_tags"])),
                "notes": _clean_column(mapping_row["notes"])
                if not pd.isna(mapping_row["notes"])
                else "",
                "alignment_method": "attack_label_contiguous_interval_to_mapping_fixed_order",
                "included": included,
                "selection_reason": "all_root_tags_resolved_to_wadi_columns" if included else "",
                "exclusion_reason": exclusion_reason,
                "unresolved_root_tags": ";".join(unresolved),
            }
        )
    return pd.DataFrame(rows), catalog


def read_numeric_process_data(
    path: Path,
    header_row: int,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Read the process variables, applying numerization and missing-value imputation per the existing loader rules."""
    raw_columns = list(pd.read_csv(path, header=header_row, nrows=0).columns)
    raw_by_clean = {_clean_column(column): str(column) for column in raw_columns}
    missing = [name for name in feature_names if name not in raw_by_clean]
    if missing:
        raise ValueError(f"过程数据缺少变量：{missing[:10]}")

    frame = pd.read_csv(
        path,
        header=header_row,
        usecols=[raw_by_clean[name] for name in feature_names],
        low_memory=False,
    )
    frame.columns = [_clean_column(column) for column in frame.columns]
    frame = frame.loc[:, list(feature_names)]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    frame = frame.fillna(frame.median(axis=0, numeric_only=True)).fillna(0.0)
    return frame.to_numpy(dtype=np.float32, copy=True)


def infer_stage(name: str) -> str:
    match = re.match(r"^([12])_", name.upper())
    return f"S{match.group(1)}" if match else "plant"


def infer_role(name: str) -> str:
    upper = name.upper()
    if upper.endswith("_SP"):
        return "setpoint"
    if upper.endswith("_CO") or "_MV_" in upper or "_MCV_" in upper:
        return "actuator"
    if upper.endswith("_STATUS") or "START_STOP" in upper:
        return "operational_state"
    if upper.endswith("_AL") or upper.endswith("_AH") or "_LS_" in upper:
        return "alarm_or_switch"
    if upper.endswith("_PV"):
        return "sensor"
    return "process_variable"


def describe_variable(name: str) -> str:
    upper = name.upper()
    meanings = (
        ("AIT", "water-quality analyzer measurement"),
        ("FIT", "flow measurement"),
        ("PIT", "pressure measurement"),
        ("LT", "tank level measurement"),
        ("MCV", "modulating control-valve command"),
        ("MV", "motorized-valve state"),
        ("P_", "pump operating state"),
        ("PIC", "pressure-control setpoint"),
        ("LS", "level-switch alarm"),
    )
    meaning = next((text for token, text in meanings if token in upper), "WADI process signal")
    return f"{meaning}; subsystem={infer_stage(name)}; role={infer_role(name)}"


def enrich_catalog_types(catalog: List[Dict[str, Any]], normal_data: np.ndarray) -> None:
    """Use the normal data's low-cardinality rule to distinguish continuous variables from discrete state variables."""
    if normal_data.shape[1] != len(catalog):
        raise ValueError("变量目录与正常数据维度不一致。")
    for index, item in enumerate(catalog):
        unique_count = int(len(np.unique(np.round(normal_data[:, index], 6))))
        item["unique_count"] = unique_count
        item["var_type"] = "state" if unique_count <= 3 else "continuous"
        item["unit_or_semantics"] = item["role"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_semicolon(value: Any) -> List[str]:
    return [item for item in str(value or "").split(";") if item]


def sampled_context(
    episode: pd.Series,
    sampled_source_rows: np.ndarray,
    context_radius: int,
) -> Tuple[int, int, int]:
    """Return the closed interval and local center index centered on the first downsampled attack point."""
    attack_positions = np.flatnonzero(
        (sampled_source_rows >= int(episode["attack_row_start"]))
        & (sampled_source_rows <= int(episode["attack_row_end"]))
    )
    if len(attack_positions) == 0:
        raise ValueError(f"{episode['episode_id']} 下采样后没有攻击点。")
    center = int(attack_positions[0])
    start = max(0, center - context_radius)
    end = min(len(sampled_source_rows) - 1, center + context_radius)
    return start, end, center - start


def select_prompt_indices(length: int, maximum: int = 21) -> List[int]:
    if length <= maximum:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, maximum, dtype=int).tolist()))


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
