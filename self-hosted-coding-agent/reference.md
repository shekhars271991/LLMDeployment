[← Back to index](PLAN.md)

## 2. Reference: platform, hardware, and model

Shared context the phases draw on. This section is deliberately reference-like and compact — the
**phases** ([see the phased roadmap, starting at Phase 1](phase-1-single-gpu-mvp/README.md)) **carry the action**.

### 2.1 Platform: self-managed vLLM/SGLang on EC2 (not SageMaker)

Self-managed EC2 is chosen over a managed endpoint because the project's whole point is turning
optimization knobs that managed endpoints don't expose (speculative-decoding configs, KV/quant
precision, prefix-cache and scheduler tuning, prefill/decode disaggregation, custom routers), it
avoids the managed-endpoint premium, and it matches the repo's existing EC2 tooling. SageMaker (or
EKS + KServe) is noted only as a later productionization path once a winning config is frozen.

### 2.2 Hardware baseline and the precision-vs-context budget

Each node is **one L40S with 48 GB** (`g6e.2xlarge`, ~$2/hr on-demand; confirm current pricing and
use spot/savings plans for long sweeps). That single number drives most early decisions, because
precision sets the KV/context budget.

Per-node VRAM math for `Qwen3.6-35B-A3B` (35B params):

| Precision | Approx. weights | KV + activation headroom on 48 GB | Feasible on 1× L40S? |
| --- | --- | --- | --- |
| BF16 | ~70 GB | — | **No** (needs TP≥2 / a multi-GPU box) |
| FP8 | ~35 GB | ~10–13 GB | Yes — modest KV/context |
| AWQ/GPTQ INT4 | ~18–20 GB | ~26–30 GB | Yes — best long-context headroom |

- **BF16 is off the table on a single-GPU node** — it only appears once we move to a multi-GPU box.
  The realistic single-GPU options are **FP8** and **INT4**.
- **Precision sets the context/KV budget.** FP8 gives near-lossless quality but little KV room; INT4
  frees ~2× the KV, which matters because coding contexts are long and reasoning traces inflate KV
  usage 2–10× (see [§3 in evaluation.md](evaluation.md)). **KV-cache quantization** (FP8 KV) is an independent lever on top.
- **Architectural detail:** Qwen3.6-35B-A3B is a **hybrid-attention** MoE — most layers use Gated
  DeltaNet (linear attention, ~constant per-token state) and only a subset use full Gated Attention,
  so KV grows **sub-linearly** vs a full-attention 35B, stretching effective context in the L40S's
  limited headroom. Confirm the real KV footprint empirically.

### 2.3 Starting model and weight variants: `Qwen/Qwen3.6-35B-A3B`

- **Why:** ~73% SWE-bench Verified from a 35B-total / ~3B-active MoE — competitive with models 3–6×
  larger, and it fits one L40S at FP8/INT4. Apache-2.0. Purpose-built for agentic coding and
  repository-level reasoning.
- **Reasoning is a baseline requirement:** Qwen3.6 emits `<think>` traces and its SWE-bench score
  **assumes reasoning on**; it also supports `preserve_thinking` (retain reasoning across turns).
  Reasoning-effort then becomes an optimization ([see Phase 2](phase-2-single-node-optimizations/README.md)) + routing ([see Phase 4](phase-4-routing/README.md)) lever.
- **Context:** 262,144 tokens native, extensible toward ~1M with YaRN — but usable context on 48 GB
  is bounded by the precision/KV budget above, not the model cap.
- **Native extras:** native MTP speculative decoding (`qwen3_next_mtp`) and Qwen3 reasoning/tool
  parsers (see [§3.3 in evaluation.md](evaluation.md)).
- **Weight variants to sweep:**
  - `Qwen/Qwen3.6-35B-A3B` — base (BF16; multi-GPU box only).
  - `Qwen/Qwen3.6-35B-A3B-FP8` — official FP8, near-lossless, single-L40S baseline.
  - `QuantTrio/Qwen3.6-35B-A3B-AWQ` (or equivalent) — community AWQ INT4 for max KV headroom.
  - `nvidia/Qwen3.6-35B-A3B-NVFP4` — NVFP4 (relevant if/when on Blackwell-class hardware).

**Caveat (stated in the report):** SWE-bench numbers are harness- and effort-sensitive and often
self-reported. Cross-model comparisons are only fair on an *identical* scaffold — which is exactly
why this project re-runs SWE-bench in-house (see [§3 in evaluation.md](evaluation.md)).

### 2.4 Scale-up ladder (models + topology)

Prove the harness + optimizations + routing on the 35B first; then add GPUs and swap in larger models
on the same rig without re-architecting.

Models (as GPUs are added):

| Tier | Model | ~SWE-bench Verified | Footprint | Notes |
| --- | --- | --- | --- | --- |
| Start | Qwen3.6-35B-A3B | ~73% | 1 GPU (FP8/INT4) | The single-GPU baseline |
| Mid | MiMo-V2-Flash (109B) | ~73% | 3–4 GPUs | Larger MoE |
| Mid | Qwen3.5-122B-A10B | ~72% | 3–4 GPUs (fits 2× H100 at ~160 GB) | |
| Top-of-footprint | DeepSeek V4 Flash | ~79% | 4+ GPUs, cheap-to-run MoE | Quality ceiling within budget |

**Out-of-footprint references** (cited, not hosted): MiniMax M2.5 (~80%), GLM-4.7 (~73%),
Kimi K2.6, DeepSeek V3.2.

Topology:

1. **Now:** 1× `g6e.2xlarge` (Phase 1) → 2× `g6e.2xlarge` for HA (Phase 3).
2. **+capacity / bigger model:** single `g6e.12xlarge` (4× L40S, 192 GB, TP within one box) or
   `g6e.48xlarge` (8× L40S) — enables BF16 / larger MoE / tensor parallelism, but a single box is a
   SPOF.
3. **HA + disaggregation:** 4 nodes (2 prefill + 2 decode + router).
4. **Realistic P/D + big models:** `p5.48xlarge` class with NVLink/EFA for the disaggregation
   speedup measurement and the top-of-footprint models.

> Detailed 2-node / HA / prefill-decode / RDMA discussion lives in **[Phase 3](phase-3-two-node-ha/README.md)**, not here.

---

## References (verify at implementation time)

- Model: `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.6-35B-A3B-FP8`, `QuantTrio/Qwen3.6-35B-A3B-AWQ`,
  `nvidia/Qwen3.6-35B-A3B-NVFP4` (Hugging Face). Apache-2.0; MoE 35B/~3B active; hybrid
  Gated-DeltaNet + Gated-Attention; 262K native context; toggleable reasoning + `preserve_thinking`;
  native `qwen3_next_mtp` speculative decoding.
- Harness: `SWE-agent/mini-swe-agent` (litellm `hosted_vllm/...`; `mini-extra swebench --subset
  verified --split test`).
- Serving: vLLM (≥ 0.24 for reasoning + tool streaming parser), SGLang (RadixAttention, cache-aware
  router), production-stack / SGLang routers, NVIDIA Dynamo (KV-aware), NIXL / LMCache / UCX
  (P/D KV transfer).
- SWE-bench Verified leaderboards and coding-agent product numbers (Steel.dev, BenchLM, llm-stats,
  swebench.com) — treated as indicative, re-run in-house.
