#!/usr/bin/env python3
"""
compare_speculative.py — Compare Phase 2 control vs speculative benchmark runs.

Groups by workload cohort (high/medium/low reuse) and context length.
Reports decode speedup, TTFT, VRAM, and speculative acceptance metrics.

Run on remote box after benchmarks:
  ./compare_speculative.py

Optional:
  ./compare_speculative.py results/bench_control_*.json results/bench_ngram-4_*.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PHASE2_DIR = SCRIPT_DIR.parent
REPORT_PATH = PHASE2_DIR / "phase2_speculative_results.txt"

COMMON_DIR = SCRIPT_DIR.parent.parent / "common"
sys.path.insert(0, str(COMMON_DIR))
from recording import start_recording  # noqa: E402

start_recording("compare_speculative", SCRIPT_DIR / "records")

COHORTS = ("high_reuse", "medium_reuse", "low_reuse")


def load_runs(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        data = json.loads(path.read_text())
        run = data["run"]
        label = run.get("run_label") or "unknown"
        run["_source"] = path.name
        by_label[label].append(run)
    return dict(by_label)


def median_of(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def aggregate_results(runs: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, float]]:
    """Median per (cohort, context) across one or more runs."""
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for row in run.get("results", []):
            if row.get("error"):
                continue
            key = (row.get("cohort", "unknown"), int(row["target_context_tokens"]))
            buckets[key].append(row)

    summary: dict[tuple[str, int], dict[str, float]] = {}
    for key, rows in buckets.items():
        ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms", -1) >= 0]
        decodes = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
        vrams = [r["peak_vram_mib"] for r in rows if r.get("peak_vram_mib", 0) > 0]
        completions = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
        summary[key] = {
            "n": len(rows),
            "ttft_median_ms": median_of(ttfts),
            "decode_tok_s_median": median_of(decodes),
            "peak_vram_mib_max": max(vrams) if vrams else 0,
            "completion_tokens_median": median_of([float(x) for x in completions]),
        }
    return summary


def aggregate_spec_metrics(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    draft_tokens: list[float] = []
    accepted: list[float] = []
    acceptance_rates: list[float] = []
    mean_lengths: list[float] = []

    for run in runs:
        delta = run.get("metrics_delta") or {}
        dt = delta.get("vllm:spec_decode_num_draft_tokens_total")
        ac = delta.get("vllm:spec_decode_num_accepted_tokens_total")
        if dt and dt > 0:
            draft_tokens.append(float(dt))
            if ac is not None:
                accepted.append(float(ac))
        ar = delta.get("acceptance_rate")
        if ar is not None:
            acceptance_rates.append(float(ar))
        mal = delta.get("mean_accepted_length")
        if mal is not None:
            mean_lengths.append(float(mal))

    return {
        "draft_tokens_median": median_of(draft_tokens),
        "accepted_tokens_median": median_of(accepted),
        "acceptance_rate_median": median_of(acceptance_rates) if acceptance_rates else None,
        "mean_accepted_length_median": median_of(mean_lengths) if mean_lengths else None,
    }


def speedup_pct(base: float, candidate: float) -> float | None:
    if base <= 0 or candidate <= 0:
        return None
    return ((candidate - base) / base) * 100.0


def discover_bench_files(extra: list[str]) -> list[Path]:
    if extra:
        return [Path(p) for p in extra]
    return sorted(RESULTS_DIR.glob("bench_*.json"))


def format_comparison(
    control_label: str,
    control_runs: list[dict[str, Any]],
    all_runs: dict[str, list[dict[str, Any]]],
) -> str:
    control_agg = aggregate_results(control_runs)
    lines = [
        "=== Phase 2 speculative decoding comparison ===",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Control label: {control_label} ({len(control_runs)} run(s))",
        "",
        "Decode speedup = (spec_median - control_median) / control_median * 100",
        "Positive speedup means higher tok/s (faster decode).",
        "",
    ]

    spec_labels = sorted(label for label in all_runs if label != control_label)
    if not spec_labels:
        lines.append("No speculative runs found to compare.")
        return "\n".join(lines)

    for spec_label in spec_labels:
        spec_runs = all_runs[spec_label]
        spec_agg = aggregate_results(spec_runs)
        spec_metrics = aggregate_spec_metrics(spec_runs)
        lines.extend(
            [
                f"--- mode: {spec_label} ({len(spec_runs)} run(s)) ---",
                f"Spec metrics (median across runs): "
                f"draft_tokens={spec_metrics['draft_tokens_median']:.0f} "
                f"accepted={spec_metrics['accepted_tokens_median']:.0f} "
                f"acceptance_rate={spec_metrics['acceptance_rate_median']} "
                f"mean_accepted_length={spec_metrics['mean_accepted_length_median']}",
                "",
                f"{'cohort':<14} {'ctx':>6} {'ctrl_dec':>10} {'spec_dec':>10} "
                f"{'speedup%':>10} {'accept':>8} {'ctrl_ttft':>10} {'spec_ttft':>10}",
            ]
        )
        for cohort in COHORTS:
            for ctx in (256, 2048, 8000):
                key = (cohort, ctx)
                base = control_agg.get(key)
                spec = spec_agg.get(key)
                if not base or not spec:
                    continue
                sp = speedup_pct(base["decode_tok_s_median"], spec["decode_tok_s_median"])
                sp_text = f"{sp:+.1f}" if sp is not None else "n/a"
                lines.append(
                    f"{cohort:<14} {ctx:>6} "
                    f"{base['decode_tok_s_median']:>10.1f} {spec['decode_tok_s_median']:>10.1f} "
                    f"{sp_text:>10} "
                    f"{spec_metrics['acceptance_rate_median']!s:>8} "
                    f"{base['ttft_median_ms']:>10.0f} {spec['ttft_median_ms']:>10.0f}"
                )
        lines.append("")

    lines.extend(
        [
            "INTERPRETATION",
            "- High-reuse cohort: n-gram should show the strongest acceptance and speedup.",
            "- Low-reuse cohort: acceptance and speedup should be weakest; slowdown is valid.",
            "- If acceptance is high but speedup is flat/negative, verification overhead dominates.",
            "- Compare medians across 3 runs per mode before drawing conclusions.",
            "",
            "Source files:",
        ]
    )
    for label, runs in sorted(all_runs.items()):
        for run in runs:
            lines.append(f"  {label}: {run.get('_source', 'unknown')}")
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    paths = discover_bench_files(args)
    if not paths:
        print("ERROR: No bench_*.json files found.", file=sys.stderr)
        return 1

    all_runs = load_runs(paths)
    control_label = "control"
    if control_label not in all_runs:
        # Fall back to first label alphabetically containing 'control' or earliest file
        for label in sorted(all_runs):
            if "control" in label:
                control_label = label
                break
        else:
            control_label = sorted(all_runs)[0]
            print(f"WARNING: no 'control' label; using {control_label}", file=sys.stderr)

    report = format_comparison(control_label, all_runs[control_label], all_runs)
    print(report)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"compare_{ts}.txt"
    out.write_text(report + "\n")
    REPORT_PATH.write_text(report + "\n")
    print(f"\nWrote {out}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
