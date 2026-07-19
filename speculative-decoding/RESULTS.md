# Results — Qwen3-32B AWQ on vLLM (L40S)

Single NVIDIA L40S (48 GB), vLLM 0.25.1, TP=1, max len 16,384, prefix caching disabled,
greedy decoding, sequential (concurrency 1). Input lengths 256 / 2,048 / 8,000 tokens.
All Phase 2 numbers are medians of 3 runs (9 prompts × 3 cohorts).

Detailed reports: [Phase 1 baseline](phase1/phase1_baseline_results.txt) ·
[Phase 2 n-gram](phase2/phase2_ngram_results.txt) ·
[Phase 2 draft model](phase2/phase2_draft_results.txt) ·
[Phase 2 comparison](phase2/phase2_speculative_results.txt)

## Phase 1 — Baseline

| Input | Median TTFT | Median decode | Peak VRAM |
|------:|------------:|--------------:|----------:|
| 256   | 102.6 ms    | 37.1 tok/s    | 42,927 MiB |
| 2,048 | 755.1 ms    | 36.2 tok/s    | 43,127 MiB |
| 8,000 | 3,363.6 ms  | 33.6 tok/s    | 43,127 MiB |

- TTFT scales with input length (prefill-bound); decode degrades only slightly.
- Prefix caching invalidates a cold-prefill benchmark, and char counts ≠ token counts —
  the corrected baseline disables caching and pads to exact tokens via `/tokenize`.

## Phase 2 — Speculative Decoding

Decode speedup vs the Phase 1-style control, by aggregate input length. The token
columns are **decode speedup %** (higher tok/s than control; e.g. `+101.7%` ≈ 2× faster
decode). "Acceptance" is the share of drafted tokens the target model accepted.

| Mode | Acceptance | Decode speedup @ 256 tok | @ 2,048 tok | @ 8,000 tok |
|------|-----------:|-------------------------:|------------:|------------:|
| ngram-2 | 52.5% | +19.8% | +12.0% | +8.2% |
| ngram-4 | 37.8% | +24.5% | +14.2% | +15.2% |
| ngram-8 | 23.5% | +26.6% | +16.3% | +16.7% |
| draft (Qwen3-0.6B, 5 tok) | 45.4% | +101.7% | +105.8% | +90.3% |

Median TTFT (ms) — speculation optimizes *decode*, not prefill, so it adds a small
TTFT cost rather than reducing it:

| Mode | 256 Tokens | 2,048 Tokens | 8,000 Tokens |
|------|----:|------:|------:|
| control | 101.9 | 747.6 | 3,361.8 |
| ngram-2 | 117.6 | 751.9 | 3,352.0 |
| ngram-4 | 117.8 | 746.4 | 3,362.4 |
| ngram-8 | 117.7 | 743.9 | 3,359.8 |
| draft (Qwen3-0.6B, 5 tok) | 148.7 | 830.1 | 3,497.2 |

Best mode per workload cohort (decode change vs control):

| Cohort | 256 | 2,048 | 8,000 |
|--------|-----|-------|-------|
| high reuse (copy/code/format) | ngram-8 (+206%) | ngram-8 (+196%) | ngram-8 (+288%) |
| medium reuse (summarize/rewrite/extract) | draft (+12%) | draft (+9%) | control (best) |
| low reuse (math/creative/logic) | ngram-8 (+27%) | draft (+17%) | draft (+27%) |

### Why draft-model speedup dips at long context

The aggregate draft speedup is a **hump, not a slide** — +101.7% (256) → +105.8% (2,048)
→ +90.3% (8,000) — and the per-cohort split runs in *opposite directions*:

| Cohort | 256 | 2,048 | 8,000 |
|--------|----:|------:|------:|
| high reuse | +121.8% | +125.0% | +113.2% |
| medium reuse | +11.5% | +8.7% | **−15.4%** |
| low reuse | +3.0% | +16.6% | **+27.3%** |

Low-reuse actually *rises* with context; the aggregate only falls at 8,000 because
medium-reuse (summarize/rewrite) collapses. Two competing effects explain this:

- **Draft "context tax" (pushes speedup down everywhere).** Each iteration runs the 0.6B
  draft **5× sequentially**, then the 32B target once to verify. The AWQ target's decode
  step is dominated by reading ~18 GB of weights, so a longer KV cache adds only ~10%
  (control: 37.2 → 33.8 tok/s across 30× more context). The 0.6B draft has ~1.4 GB of
  weights, so at 8,000 tokens attention/KV reads over the long context start to dominate
  its cost — ×5 passes. The draft's per-step cost grows *proportionally faster* than the
  target's, eroding the `t_draft/t_verify` ratio that made drafting nearly free at 256.
  (N-gram has no model forward pass, so it avoids this tax — high-reuse n-gram even *rose*
  at 8,000.)
- **Acceptance is content-dependent (splits the cohorts).** For templated continuations
  (extraction, math, copy) the draft keeps predicting well even at long context, so
  speedup holds or climbs. For summarize/rewrite/creative, a 0.6B model guesses the
  continuation worse as input grows, acceptance drops, and the (now larger) draft cost is
  paid on rejected tokens — turning speedup negative.

At 8,000 tokens the bigger draft tax hits everything, and where acceptance also drops the
two effects compound (medium-reuse goes negative); where acceptance holds, the tax is
overcome and speedup even grows (low-reuse math/logic). Confirming the acceptance half
directly needs per-window acceptance data — see the follow-up under
[Validity](#follow-up-per-window-acceptance-rate).

## Key Findings

- **No single method wins everywhere.** N-gram-8 dominates repetitive/templated output;
  the draft model gives the best aggregate throughput (~+90–106%) and broad gains.
- **N-gram is workload-gated.** High-reuse prompts gain up to +288%, but medium-reuse
  prompts slow ~6–8% at every depth — speculation isn't free when proposals miss.
- **Acceptance rate ≠ throughput.** N-gram-8 had the *lowest* acceptance (23.5%) yet the
  fastest n-gram decode, because accepted runs are longer and high-reuse text benefits
  from deeper lookahead.
- **Speculation trades TTFT for decode.** N-gram adds ~16 ms TTFT (negligible at long
  contexts); the draft model adds ~47–135 ms (extra 0.6B model + V1 runner). Worth it
  only when outputs are long enough to recoup it.
- **Practical takeaway:** route by workload rather than enabling one method globally —
  n-gram for copy/code/format, draft model for extraction/arithmetic/long structured
  output, control for creative/rewrite traffic.

## Validity

405 requests total (5 modes × 3 runs × 27 requests), all completed error-free at exact
token lengths. Server launch logs were cross-checked against result labels; mislabeled
and stale-metadata files were corrected before analysis (see per-phase reports).

*Latency benchmark at concurrency 1 — isolates per-request latency, not concurrent
throughput. Synthetic prompt set; treat cohort routing as indicative, not definitive.*

### Follow-up: per-window acceptance rate

Acceptance is reported as a **single aggregate per mode** because `bench.py` snapshots
vLLM's cumulative spec-decode counters (`vllm:spec_decode_num_{draft,accepted}_tokens_total`)
only once before and once after the full run, so the 3 context lengths × 9 prompts are
summed together and can't be sliced afterward. A per-window (or per-request) acceptance
breakdown would directly test whether the draft model's decode speedup falls at 8,000
tokens because acceptance drops on medium/creative prompts vs. because draft compute
overhead grows with context. Fix for a future run: take a counter delta around each
`for ctx in context_lengths` block (counters are cumulative, so per-block deltas are exact)
and store it in a `per_context_metrics` field — leaving the current results untouched.
