#!/usr/bin/env python3
"""compare_results.py — Build the Phase 2 per-lever delta table (PLAN §3.7).

Reads every swebench_*.json summary written by 03_run_benchmark.sh, treats the
baseline config as the reference, and prints a table of each lever's SWE-bench
resolve rate and its delta vs the baseline — the "which levers were adopted and
why" deliverable that closes Phase 2.

Usage:
    python3 compare_results.py [RESULTS_DIR]

RESULTS_DIR defaults to records/results next to this script. If multiple runs
exist for the same config_name, the most recent (by timestamp_utc) wins.
"""
from __future__ import annotations

import glob
import json
import os
import sys

BASELINE_HINT = "baseline"


def load_summaries(results_dir: str) -> list[dict]:
    summaries: dict[str, dict] = {}
    for path in glob.glob(os.path.join(results_dir, "swebench_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
            continue
        name = data.get("config_name") or os.path.basename(path)
        data["_path"] = path
        # Keep the most recent summary per config_name.
        prev = summaries.get(name)
        if prev is None or (data.get("timestamp_utc", "") > prev.get("timestamp_utc", "")):
            summaries[name] = data
    return list(summaries.values())


def pick_baseline(rows: list[dict]) -> dict | None:
    for row in rows:
        if BASELINE_HINT in (row.get("config_name") or ""):
            return row
    return rows[0] if rows else None


def fmt(value, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value}{suffix}"


def main() -> int:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "records", "results"
    )
    if not os.path.isdir(results_dir):
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        return 1

    rows = load_summaries(results_dir)
    if not rows:
        print(f"No swebench_*.json summaries found under {results_dir}.")
        return 0

    # Sort by sub-phase then config_name for a stable, readable order.
    rows.sort(key=lambda r: (r.get("subphase", ""), r.get("config_name", "")))

    baseline = pick_baseline(rows)
    base_rate = baseline.get("resolve_rate_pct") if baseline else None
    base_name = baseline.get("config_name") if baseline else "(none)"

    header = (
        f"{'sub':<4} {'config':<24} {'engine':<7} {'subset':<9} "
        f"{'resolved':>9} {'rate%':>7} {'Δ vs base':>10}"
    )
    print(f"Results dir : {results_dir}")
    print(f"Baseline    : {base_name} ({fmt(base_rate, '%')})")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for row in rows:
        cfg = row.get("config", {}) or {}
        rate = row.get("resolve_rate_pct")
        if rate is not None and base_rate is not None:
            delta = f"{rate - base_rate:+.2f}"
        else:
            delta = "-"
        resolved = f"{fmt(row.get('resolved'))}/{fmt(row.get('instances_attempted'))}"
        print(
            f"{row.get('subphase',''):<4} "
            f"{(row.get('config_name') or '')[:24]:<24} "
            f"{str(cfg.get('engine',''))[:7]:<7} "
            f"{str(row.get('subset',''))[:9]:<9} "
            f"{resolved:>9} "
            f"{fmt(rate):>7} "
            f"{delta:>10}"
        )

    print("-" * len(header))
    print("Adopt a lever only if Δ vs base is a clear win (quality) — and cross-check")
    print("its serving-metric / cost deltas from the serve logs before adopting (PLAN §3.7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
