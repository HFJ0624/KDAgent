"""Completeness checker script for the parsed-results CSV.

This script is used to quickly verify, after a run, that every record in
``{model}_parsed.csv`` has the key fields required for evaluation populated:

* ``gt_vars``: the ground-truth root cause variables;
* ``top10_vars``: the Top-10 candidate variables;
* ``primary_root_cause``: the primary root cause prediction;
* ``predicted_root_causes``: the complete prediction list.

The script prints the field-completeness check result for each case and, when a field is missing,
prints a ``missing`` line to help quickly locate the anomalous case, providing a basis for a later
rerun or human investigation.

Typical usage (after the experiment finishes):
    python scripts/verify_csv.py
"""

import csv
import json
import sys

# Path of the parsed-results CSV to verify (hardcoded for easy double-click / command-line runs)
path = r"d:\workspace\lab_TARCA_S2S\llm_rca_experiment\outputs_smoke_real\parsed_results\qwen-plus_parsed.csv"

# Read with utf-8-sig so the file opens cleanly in Excel without mojibake
with open(path, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"Row count: {len(rows)}")

# Check row by row whether the key fields are non-empty; summarize each row's status with ALL_FIELDS_POPULATED
for r in rows:
    # A row is only considered "complete" when all key fields are non-empty
    ok = all([r.get("gt_vars"), r.get("top10_vars"), r.get("primary_root_cause"), r.get("predicted_root_causes")])
    print(f"case_id={r.get('case_id')} gt={r.get('gt_vars')} top10={str(r.get('top10_vars',''))[:60]} primary={r.get('primary_root_cause')} predicted={r.get('predicted_root_causes')} -> ALL_FIELDS_POPULATED={ok}")
    if not ok:
        # When missing, print each field's populated status to locate which one is absent
        print(f"  missing: gt={bool(r.get('gt_vars'))}, top10={bool(r.get('top10_vars'))}, primary={bool(r.get('primary_root_cause'))}, predicted={bool(r.get('predicted_root_causes'))}")
