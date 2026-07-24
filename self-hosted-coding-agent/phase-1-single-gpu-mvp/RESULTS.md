# Phase 1 — SWE-bench Results (Single-GPU MVP)

> **Status: VALID SMOKE RUN.** 5-instance SWE-bench Lite smoke on a single L40S — used to prove
> the pipeline and the harness fix, **not** a statistically meaningful score. Scale to full Lite
> (300) / Verified for the headline baseline. An earlier attempt failed on a harness bug; it is
> archived at the bottom.

## Headline (Attempt 2 — after fix)

- **Resolve rate: 80.0% — 4 resolved / 5 attempted** (SWE-bench Lite, 5-instance smoke; `dataset_total = 300`).
- Cost-tracking `RuntimeError` is **gone** (`MSWEA_COST_TRACKING=ignore_errors`); run finished cleanly (exit 0).
- **4 of 5** predictions produced non-empty `model_patch`; the 1 failure is a genuine
  **context-window overflow**, not the old bug.
- ⚠️ 5 instances is a smoke test to validate wiring — treat 80% as "pipeline works", not as the
  model's SWE-bench Lite score.

## Run metadata

| Field | Value |
|-------|-------|
| Date (UTC) | 2026-07-20 |
| Run id | `phase1` (agent run `run_20260720T075631Z`) |
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` (served as `qwen3.6-35b-a3b-fp8`) |
| Instance | `g6e.2xlarge` (1× NVIDIA L40S 48 GB), us-east-1 |
| Serving engine | vLLM, reasoning ON (`--reasoning-parser qwen3`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_xml`, `--enable-prefix-caching`) |
| Serving config | `max-model-len=32768`, `gpu-memory-utilization=0.92`, `max-num-seqs=16`, TP=1 |
| Harness | mini-swe-agent 2.4.5 → SWE-bench Lite (Docker evaluation), `MSWEA_COST_TRACKING=ignore_errors` |
| Slice | 5-instance smoke (`INSTANCE_LIMIT=5`), `workers=4` |
| Dataset | `princeton-nlp/SWE-bench_Lite` |
| Results folder | `downloaded-results/rerun_20260720T080249Z/` (summary `records/results/swebench_20260720T075631Z.json`) |

## Per-instance outcome

| Instance | Exit status | Patch | Resolved |
|----------|-------------|-------|----------|
| astropy__astropy-14995 | Submitted | non-empty | ✅ |
| astropy__astropy-12907 | Submitted | non-empty | ✅ |
| astropy__astropy-14365 | Submitted | non-empty | ✅ |
| astropy__astropy-6938 | Submitted | non-empty | ✅ |
| astropy__astropy-14182 | ContextWindowExceededError | empty | ❌ |

(All 5 smoke instances happen to be `astropy` — an artifact of taking the first 5 of the dataset,
not a representative sample.)

## Serving observations

- Engine init ~212 s (incl. `torch.compile` ~50 s) on first start; then "Application startup complete".
- Single-stream decode ≈ **75 tok/s** on the L40S at FP8 (from the serve log during the run).
- Non-fatal warning: no tuned FP8 kernel config for the L40S at some MoE shapes → default W8A8
  block-FP8 kernel (possible minor perf hit, not an error).

## Known issues / next steps

1. **Context window:** `astropy-14182` overflowed the served context (`max-model-len=32768`). Note
   the benchmark script's `MAX_MODEL_TOKENS=64000` **exceeds** what the server actually allows
   (32768) — misleading. Fix by either raising `--max-model-len` (KV budget permitting on 48 GB),
   lowering `MAX_MODEL_TOKENS` to match, and/or enabling agent history trimming.
2. **Scale up:** run full **SWE-bench Lite (300)**, then **Verified**, for a real baseline (the
   5-instance smoke is not statistically meaningful).
3. **Capacity:** add a concurrent load/capacity measurement (see `../evaluation.md`) alongside the
   quality number.

## Evidence (files in `downloaded-results/rerun_20260720T080249Z/`)

- Summary: `records/results/swebench_20260720T075631Z.json`
- Predictions: `records/results/run_20260720T075631Z/preds.json`
- Exit statuses: `records/results/run_20260720T075631Z/exit_statuses_*.yaml`
- Per-instance trajectories: `records/results/run_20260720T075631Z/<instance_id>/<instance_id>.traj.json`

---

## Archive — Attempt 1 (INVALID: harness cost-tracking bug)

First run (`downloaded-results/20260720T072430Z/`, agent run `run_20260720T074102Z`) reported
**0/5** — but this was a **harness misconfiguration, not a model result**. The model answered all
5 requests (200 OK), but mini-swe-agent aborted every task on its first model call with:

```
ValueError: Cost must be > 0.0, got 0.0
  → RuntimeError: Error calculating cost for model hosted_vllm/qwen3.6-35b-a3b-fp8:
    Cost must be > 0.0, got 0.0, perhaps it's not registered?
```

Cause: our litellm `registry.json` sets per-token cost to `0.0` (self-hosted = free), and
mini-swe-agent's default cost tracking treats a `0.0` cost as "unregistered" and raises, producing
empty patches. Fix: `export MSWEA_COST_TRACKING=ignore_errors` (applied in `03_run_benchmark.sh`),
which produced the valid Attempt 2 above. Infra, serving, model responses, and Docker evaluation
all worked in Attempt 1 — only the cost guard broke the task loop.
