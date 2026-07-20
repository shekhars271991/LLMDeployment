[← Back to index](../PLAN.md)

# Phase 1 — Single-GPU end-to-end MVP

*Keep this phase lean — it is the simplest of the four.*

**Goal:** stand up the *entire* pipeline once, on **ONE GPU**, and get a first recorded SWE-bench
number. This is the "prove the pipeline works" milestone — nothing more.

**Scope (do exactly this, and no more):**

- **1× `g6e.2xlarge` (1× L40S 48 GB)** — a single node, no load balancer.
- Model `Qwen/Qwen3.6-35B-A3B-FP8`, **reasoning ON** with the required parser flags (see [§3.3 in evaluation.md](../evaluation.md)).
- **vLLM**, OpenAI-compatible `/v1`.
- Wire up `mini-swe-agent` against that endpoint (see [§3.1 in evaluation.md](../evaluation.md)).
- Run **SWE-bench Lite** to iterate the wiring, then **SWE-bench Verified** for the headline number.
- Generate the report and record the baseline using the **§3.7 template** (see [evaluation.md](../evaluation.md)).

**Explicitly OUT of Phase 1** (deferred to later phases): optimization sweeps, quantization
comparisons, the second node / HA, prefill-decode disaggregation, and all routing. Keep this phase
the simplest of the four.

Illustrative single-node FP8 launch (a lever description, not a runbook):

```
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --enable-prefix-caching \
  --max-model-len <fit-to-KV-budget> --max-num-seqs <tune>
```

**Exit criteria:**

- A reproducible **end-to-end run** (endpoint → harness → scored SWE-bench).
- A **recorded baseline** (quality + serving metrics + cost proxy) in the §3.7 template (see [evaluation.md](../evaluation.md)).
- *(Optional)* one **frontier-model anchor** run through the same harness (see [§3.5 in evaluation.md](../evaluation.md)).

---
**Navigation:** [Index](../PLAN.md) · [Phase 2 →](../phase-2-single-node-optimizations/README.md)
