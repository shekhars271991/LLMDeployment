[← Back to index](../PLAN.md)

# Phase 2 — Single-node optimizations

*Sub-phases 2a–2f, applied one lever at a time.*

On the **same one GPU**, apply optimizations **one at a time**, each measured against the Phase 1
baseline on the fixed harness. **Adopt a lever only if it wins.** Each sub-phase closes with its own
§3.7 record (see [evaluation.md](../evaluation.md)) and a per-lever delta.

- **Phase 2a — Precision / KV headroom.** *Goal:* find the best quality-vs-KV point.
  FP8 vs AWQ INT4 vs FP8 KV cache. *Measure:* SWE-bench quality delta per precision vs KV/context
  gained (re-run SWE-bench per precision, don't assume parity).
- **Phase 2b — Prefix caching + chunked prefill + long context.** *Goal:* exploit the shared agentic
  prefix (system prompt + tool schema replayed every turn) and tune `--max-model-len` to real
  codebase sizes. *Measure:* TTFT win on shared prefixes; long-prompt behavior under concurrency.
- **Phase 2c — Batching / scheduler + CUDA graphs / `torch.compile`.** *Goal:* raise throughput
  without breaking latency. *Measure:* throughput/latency vs `--max-num-seqs` and scheduler settings;
  per-decode-step overhead reduction.
- **Phase 2d — Speculative decoding.** *Goal:* speed up decode (matters more here because reasoning
  makes decode dominant). Native MTP (`qwen3_next_mtp`), n-gram, draft model, EAGLE. *Measure:*
  decode speedup at fixed quality.
- **Phase 2e — Reasoning-effort sweep.** *Goal:* quantify the reasoning trade. Reasoning on vs
  off/low (`chat_template_kwargs: {enable_thinking: ...}`). *Measure:* quality vs cost/latency.
- **Phase 2f — Serving-engine A/B.** *Goal:* test whether SGLang beats vLLM on the winning config.
  SGLang **RadixAttention** (token-level prefix tree) targets the long, stable coding prefix; both
  expose OpenAI `/v1` so harness/client stay identical. *Measure:* prefill latency, prefix hit rate,
  capacity. (Triton/TensorRT-LLM deferred — AOT compilation kills experiment velocity; keep only as
  an optional final "compiled max-throughput" datapoint.)

**Exit criteria:** a recorded **"best single-node config"** plus per-sub-phase deltas showing which
levers were adopted and why.

---
**Navigation:** [← Phase 1](../phase-1-single-gpu-mvp/README.md) · [Index](../PLAN.md) · [Phase 3 →](../phase-3-two-node-ha/README.md)
