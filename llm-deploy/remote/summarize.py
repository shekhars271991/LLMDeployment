#!/usr/bin/env python3
"""
summarize.py — Aggregate results/bench_*.json into a readable baseline table.

Prerequisite: bench.py has written at least one results/bench_*.json
Run on remote box: python summarize.py [optional/path/to/bench.json]

Prints table to stdout and writes results/summary_<timestamp>.txt
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def latest_bench_file() -> Path | None:
    files = sorted(RESULTS_DIR.glob("bench_*.json"))
    return files[-1] if files else None


def gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        )
        return "; ".join(line.strip() for line in out.strip().splitlines())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def vllm_version() -> str:
    try:
        import vllm  # noqa: PLC0415

        return vllm.__version__
    except Exception:  # noqa: BLE001
        return "unknown"


def load_config_snapshot() -> dict[str, str]:
    cfg: dict[str, str] = {}
    env_path = SCRIPT_DIR / "config.env"
    if not env_path.exists():
        return cfg
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def aggregate_by_context(results: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    by_ctx: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        if r.get("error"):
            continue
        ctx = int(r["target_context_tokens"])
        by_ctx.setdefault(ctx, []).append(r)

    summary: dict[int, dict[str, float]] = {}
    for ctx, rows in sorted(by_ctx.items()):
        ttfts = [x["ttft_ms"] for x in rows if x.get("ttft_ms", -1) >= 0]
        decodes = [x["decode_tok_s"] for x in rows if x.get("decode_tok_s")]
        vrams = [x["peak_vram_mib"] for x in rows if x.get("peak_vram_mib", 0) > 0]

        summary[ctx] = {
            "n": len(rows),
            "ttft_mean_ms": statistics.mean(ttfts) if ttfts else 0,
            "ttft_median_ms": statistics.median(ttfts) if ttfts else 0,
            "decode_tok_s_mean": statistics.mean(decodes) if decodes else 0,
            "decode_tok_s_median": statistics.median(decodes) if decodes else 0,
            "peak_vram_mib_max": max(vrams) if vrams else 0,
        }
    return summary


def format_table(
    meta: dict[str, Any],
    agg: dict[int, dict[str, float]],
) -> str:
    lines = [
        "=== Qwen3-32B AWQ baseline summary ===",
        f"Timestamp (UTC): {meta.get('timestamp', '')}",
        f"Model: {meta.get('model', '')}",
        f"vLLM: {meta.get('vllm_version', '')}",
        f"GPU: {meta.get('gpu', '')}",
        f"TP: {meta.get('config', {}).get('TP')}  MAXLEN: {meta.get('config', {}).get('MAXLEN')}  "
        f"GPU_MEM_UTIL: {meta.get('config', {}).get('GPU_MEM_UTIL')}",
        "",
        f"{'ctx_tokens':>10} {'n':>4} {'TTFT_mean':>10} {'TTFT_med':>10} "
        f"{'dec_mean':>10} {'dec_med':>10} {'VRAM_max':>10}",
        f"{'':>10} {'':>4} {'(ms)':>10} {'(ms)':>10} {'(tok/s)':>10} {'(tok/s)':>10} {'(MiB)':>10}",
    ]
    for ctx, s in sorted(agg.items()):
        lines.append(
            f"{ctx:>10} {int(s['n']):>4} "
            f"{s['ttft_mean_ms']:>10.1f} {s['ttft_median_ms']:>10.1f} "
            f"{s['decode_tok_s_mean']:>10.1f} {s['decode_tok_s_median']:>10.1f} "
            f"{s['peak_vram_mib_max']:>10.0f}"
        )
    lines.append("")
    lines.append("Paste this block into RUNBOOK.md → Results section.")
    return "\n".join(lines)


def main() -> int:
    bench_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_bench_file()
    if not bench_path or not bench_path.exists():
        print("ERROR: No bench_*.json found in results/. Run bench.py first.", file=sys.stderr)
        return 1

    data = json.loads(bench_path.read_text())
    run = data["run"]
    results = run.get("results", [])
    agg = aggregate_by_context(results)

    cfg = load_config_snapshot()
    meta = {
        "timestamp": run.get("timestamp"),
        "model": run.get("model"),
        "vllm_version": vllm_version(),
        "gpu": gpu_info(),
        "config": {**run.get("config", {}), **{k: cfg.get(k) for k in ("TP", "MAXLEN", "GPU_MEM_UTIL")}},
        "source_file": str(bench_path.name),
    }

    table = format_table(meta, agg)
    print(table)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"summary_{ts}.txt"
    out.write_text(table + f"\n\nSource: {bench_path.name}\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
