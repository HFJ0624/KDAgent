"""General-purpose helper module.

This module provides basic utility functions reused across files in the LLM RCA
experiments, mainly grouped as follows:

1. Path and directory utilities: :func:`ensure_dir` ensures a directory exists;
2. Logging utilities: :func:`setup_logger` creates a Logger that writes to both
   stdout and a timestamped log file;
3. JSONL / JSON / CSV file I/O: :func:`load_jsonl`,
   :func:`append_jsonl`, :func:`write_json`, :func:`read_json`,
   :func:`write_csv`, :func:`write_df_csv`;
4. YAML reading: :func:`read_yaml` (with a ``_minimal_yaml_load``
   fallback, enabling basic parsing when PyYAML is not installed);
5. Other helpers: :func:`get_api_key`, :func:`chunked`,
   :func:`sanitize_filename`.

The design principle of this module is "pure functions + no side effects": apart
from file I/O it keeps no global state, which eases unit testing and reuse.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def ensure_dir(path: str) -> str:
    """Ensure that the directory exists, creating it automatically if missing; returns the directory path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(
    log_dir: str = "logs",
    name: str = "rca_experiment",
    log_filename: Optional[str] = None,
    model_name: Optional[str] = None,
    use_rag: Optional[bool] = None,
) -> logging.Logger:
    """Configure a logger that writes to both stdout and a timestamped log file.

    Args:
        log_dir: The log file directory
        name: The logger name
        log_filename: A specified log filename (optional); auto-generated if not given
        model_name: The model name (optional), used in the log format
        use_rag: Whether RAG is used (optional), used in the log format
    """
    ensure_dir(log_dir)
    logger = logging.getLogger(name)

    if not log_filename:
        model_part = sanitize_filename(model_name) if model_name else "rca_experiment"
        rag_part = "with-rag" if use_rag else "no-rag"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"rca_experiment_{model_part}_{rag_part}_{timestamp}.log"

    if logger.handlers:
        for handler in list(logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    if model_name:
        model_tag = sanitize_filename(model_name)
        rag_tag = "with-rag" if use_rag else "no-rag"
        fmt_str = f"[%(asctime)s][%(levelname)s][{model_tag}][{rag_tag}] %(message)s"
    else:
        fmt_str = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"

    fmt = logging.Formatter(
        fmt_str,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_file = os.path.join(log_dir, log_filename)
    ensure_dir(os.path.dirname(log_file))
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write a dict object to a JSON file with UTF-8 encoding.

    Creates the target directory automatically if missing; writes with
    ``ensure_ascii=False`` and ``indent=2`` to keep Chinese characters and
    ensure good readability.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    """Read a JSON file with UTF-8 encoding and return the parsed dict object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    import csv

    rows = list(rows)
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        # QUOTE_MINIMAL -> pandas/Excel/Numbers all parse it reliably;
        # fields containing commas/newlines/quotes are correctly wrapped, avoiding downstream misalignment.
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_df_csv(path: str, df) -> None:
    """Write a pandas DataFrame or a list of dicts to CSV (supports utf-8-sig Chinese encoding)."""
    ensure_dir(os.path.dirname(path))
    try:
        # Duck typing: assume it is a pandas DataFrame
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except AttributeError:
        # Fall back to the csv module (here df is actually a list of dicts)
        if not df:
            return
        rows = list(df)
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        write_csv(path, rows, fieldnames=fieldnames)


def read_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML file and return the parsed dict.

    Prefers PyYAML's ``safe_load``; if PyYAML is not installed or parsing fails,
    falls back to this file's built-in minimal parser ``_minimal_yaml_load``,
    which is sufficient for the top-level keys and list-of-dicts structure needed
    by ``models.yaml``.
    """
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return _minimal_yaml_load(path)


def _minimal_yaml_load(path: str) -> Dict[str, Any]:
    """A minimal YAML parser.

    Only supports the subset syntax needed by this project's ``models.yaml``:
    top-level key-value pairs and list items of dicts starting with ``-``; supports
    automatic conversion of primitive types such as string, integer, float,
    boolean (true/false/yes/no), and null.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.readlines()]

    root: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    current_list: List[Dict[str, Any]] = []
    current_item: Optional[Dict[str, Any]] = None

    def _coerce(v: str) -> Any:
        """Automatically convert a string value into an appropriate Python type.

        Processing order: strip quotes -> boolean -> null -> float -> int -> original string.
        """
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        if v.lower() in ("true", "yes"):
            return True
        if v.lower() in ("false", "no"):
            return False
        if v.lower() in ("null", "~", ""):
            return None
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    for raw in lines:
        # Strip an inline # comment before judging (simple handling; does not consider # inside strings)
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)

        if indent == 0 and stripped.endswith(":"):
            # Top-level list key: e.g. ``models:``
            current_list_key = stripped[:-1].strip()
            current_list = []
            current_item = None
            root[current_list_key] = current_list
            continue

        if stripped.startswith("- "):
            # A new list item starts: push the previous item into the list
            if current_item is not None:
                current_list.append(current_item)
            current_item = {}
            rest = stripped[2:].strip()
            if ":" in rest:
                k, v = rest.split(":", 1)
                current_item[k.strip()] = _coerce(v)
            continue

        if ":" in stripped:
            k, v = stripped.split(":", 1)
            key = k.strip()
            val = _coerce(v)
            if indent == 0:
                # Top-level scalar key
                root[key] = val
            else:
                # Indented key inside a list item
                if current_item is None:
                    current_item = {}
                current_item[key] = val

    if current_item is not None:
        current_list.append(current_item)
    return root


def get_api_key(env_var: str) -> Optional[str]:
    """Read the API Key from an environment variable; returns None when not set."""
    return os.environ.get(env_var)


def chunked(seq: List[Any], n: int) -> Iterable[List[Any]]:
    """A generator that splits a list into sublists of every ``n`` elements.

    Commonly used for batch API calls or batch writes.
    """
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def normalize_var_name(value: Any) -> str:
    """Canonicalize a SWaT/variable name for comparison: uppercase, strip
    surrounding whitespace, and remove all internal spaces.

    This is the single authoritative normalization used across parsing,
    validation, evaluation, and the iterative self-refinement agent so that
    equal variable IDs compare consistently regardless of casing/spacing.
    """
    return str(value).strip().upper().replace(" ", "")


def sanitize_filename(name: str) -> str:
    """Convert any string into a filename safe for both Windows and Unix.

    Rules:
      - Replace illegal characters (``<>:"/\\|?*`` and whitespace) with underscores ``_``;
      - Collapse runs of consecutive underscores into a single one;
      - Strip leading/trailing underscores and dots, avoiding rejection on Windows or misfiling as hidden files.
    """
    if name is None:
        return ""
    s = str(name)
    illegal = '<>:"/\\|?*\r\n\t'
    for ch in illegal:
        s = s.replace(ch, "_")
    s = s.replace(" ", "_")
    # Collapse consecutive underscores
    while "__" in s:
        s = s.replace("__", "_")
    # Strip leading/trailing underscores and dots (Windows safety)
    return s.strip("_.")
