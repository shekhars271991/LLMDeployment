# LLM Deployment

Benchmarking different LLM deployment methodologies and inference optimizations on
real GPU hardware — measuring how each technique affects latency, throughput, and
memory for large models served with [vLLM](https://github.com/vllm-project/vllm).

The goal is hands-on, reproducible evidence: run controlled experiments, measure
TTFT / decode throughput / VRAM under fixed conditions, and document what actually
helps for which workloads.

## Experiments

### [`speculative-decoding/`](speculative-decoding/)

Deploy **Qwen3-32B AWQ** on a single NVIDIA L40S and evaluate speculative decoding.

- **Phase 1 — Baseline:** cold-prefill latency across 256 / 2,048 / 8,000-token inputs,
  with prefix caching disabled and exact token-length padding.
- **Phase 2 — Speculative decoding:** control vs n-gram prompt lookup (depths 2/4/8)
  vs draft-model speculation (Qwen3-0.6B), split across high / medium / low token-reuse
  workloads.

See [`speculative-decoding/RESULTS.md`](speculative-decoding/RESULTS.md) for the
combined findings.

## Highlights so far

- No single speculative method wins everywhere — gains are strongly workload-dependent.
- N-gram lookup gave up to **+288%** decode throughput on repetitive/templated output,
  but slowed medium-reuse prompts ~6–8%.
- Draft-model speculation delivered the best aggregate decode throughput (**~+90–106%**)
  at the cost of higher TTFT.
- Acceptance rate alone does not predict throughput; TTFT-vs-decode trade-offs and
  output length matter.

## Method

- Fixed serving config per experiment (model, TP, max len, GPU memory utilization).
- Exact input token lengths verified via vLLM's `/tokenize` endpoint.
- Prefix caching disabled for clean cold-prefill measurements.
- Medians reported across repeated runs; server configs cross-checked against results.

Each experiment folder is self-contained with its own scripts, runbooks, config
templates, and results.
