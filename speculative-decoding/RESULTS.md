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

Decode speedup vs the Phase 1-style control, by aggregate input length:

| Mode | Acceptance | 256 | 2,048 | 8,000 |
|------|-----------:|----:|------:|------:|
| ngram-2 | 52.5% | +19.8% | +12.0% | +8.2% |
| ngram-4 | 37.8% | +24.5% | +14.2% | +15.2% |
| ngram-8 | 23.5% | +26.6% | +16.3% | +16.7% |
| draft (Qwen3-0.6B, 5 tok) | 45.4% | +101.7% | +105.8% | +90.3% |

Median TTFT (ms) — speculation optimizes *decode*, not prefill, so it adds a small
TTFT cost rather than reducing it:

| Mode | 256 | 2,048 | 8,000 |
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
