# Self-Hosted Coding Agent — Experimentation Plan

**Status:** Planning only. This document defines an experimentation project; it contains no
provisioning scripts, runbooks, or execution steps. Illustrative serving flags appear inline to
describe *levers*, not to be run verbatim.

**Sibling project:** `../speculative-decoding/`. This project reuses that project's learnings on
vLLM, quantization, prefix caching, and benchmark methodology (fixed configs, cross-checked launch
logs, medians across repeats), and its method-routing result folds in as one routing signal.

---

## 0. At a glance

Find the **best-fit self-hosted deployment for a coding agent** and prove it with SWE-bench.

- **Now organized as a phased roadmap.** Each phase is **end-to-end** and produces a **recorded
  report** (using the [§3 template in evaluation.md](evaluation.md)) *before* the next phase begins. Complexity deliberately lives in
  later phases; Phase 1 is kept lean.
- **Phase 1 — single-GPU end-to-end MVP:** stand up the whole pipeline once on **one L40S** and get a
  first recorded SWE-bench number.
- **Phase 2 — single-node optimizations (2a/2b/…):** on the same one GPU, sweep precision, prefix
  caching, batching, speculative decoding, reasoning effort, and engine A/B — one lever at a time.
- **Phase 3 — two-node HA + its optimizations:** add a second node for HA, then the serving-routing
  ladder and a scoped prefill/decode disaggregation experiment.
- **Phase 4 — routing + its optimizations:** the application-routing layer (cascade with an
  executable verifier), combined with the best serving config into a cost/quality/capacity frontier.
- **Model:** `Qwen/Qwen3.6-35B-A3B` (MoE 35B total / ~3B active, Apache-2.0, ~73% SWE-bench Verified,
  reasoning-enabled by default), starting on **1× NVIDIA L40S 48 GB**.
- **Deliverable:** a recorded report per phase plus a final **best-fit recommendation** — *"this
  config resolves X% of SWE-bench Verified"* and *"it serves ~N concurrent users at p95 TTFT <
  target"*, positioned on a quality-vs-cost frontier against commercial coding agents.

---

## 1. Objective and success criteria

**Objective.** Determine the configuration (model + precision + serving engine + optimization set +
routing policy) that maximizes **SWE-bench Verified resolve rate per GPU-dollar** while meeting
interactive latency SLOs, deployed self-hosted on AWS EC2, and usable from a local coding agent
(e.g. an OpenRouter-/OpenAI-compatible client).

**Success criteria.**

1. **Quality:** a reproducible SWE-bench Verified resolve rate on a fixed in-house harness, within a
   documented gap of a frontier commercial model run through the *same* harness.
2. **Capacity:** a measured concurrent-user capacity at defined p95 TTFT / inter-token-latency SLOs,
   including on the 2-node HA rig and in degraded (1-node-down) mode.
3. **Cost:** a blended GPU-$/resolved-task figure, compared against commercial API $/task.
4. **Best-fit statement:** a single recommended config plus a routing policy, with the evidence
   (quality, capacity, cost) that justifies it, and a scale-up recommendation.

**Non-goals.** Training/fine-tuning the model; a production-hardened control plane; multi-region;
managed-endpoint (SageMaker) productionization (noted only as a later option).

---

## Document map

This plan is split across the files below. **The phased roadmap is the backbone.** Each phase is
end-to-end, has explicit **Exit criteria**, and produces a recorded report using the
[§3.7 template in evaluation.md](evaluation.md) before moving on. Later phases inherit the winning config of the
prior phase.

- **[reference.md](reference.md)** — platform, hardware, and model reference (§2.1–2.4: EC2 vs
  SageMaker, the precision-vs-context VRAM budget, the starting model + weight variants, and the
  scale-up ladder), plus the References list.
- **[evaluation.md](evaluation.md)** — the cross-cutting evaluation methodology (§3.1–3.7: headline
  metric + harness, reasoning-aware SLOs, required serving flags, load/capacity benchmark, industry
  baselines, validity controls, and the per-phase record template) that every phase reuses.
- **[phase-1-single-gpu-mvp.md](phase-1-single-gpu-mvp/README.md)** — Phase 1, the single-GPU end-to-end
  MVP. *Exit:* a reproducible end-to-end run and a recorded baseline (quality + serving metrics +
  cost proxy).
- **[phase-2-single-node-optimizations.md](phase-2-single-node-optimizations/README.md)** — Phase 2,
  single-node optimizations (sub-phases 2a–2f), one lever at a time. *Exit:* a recorded "best
  single-node config" plus per-sub-phase deltas showing which levers were adopted and why.
- **[phase-3-two-node-ha.md](phase-3-two-node-ha/README.md)** — Phase 3, two-node HA and its optimizations
  (HA replicas, Layer-B serving routing, scoped P/D disaggregation). *Exit:* recorded HA capacity
  numbers (per-replica / aggregate / failover) and the best 2-node serving config, plus P/D
  mechanism-vs-realistic findings.
- **[phase-4-routing.md](phase-4-routing/README.md)** — Phase 4, application (Layer-A) routing and its
  optimizations, combined with the best serving config. *Exit:* a recorded routing policy plus the
  combined cost/quality/capacity frontier.
- **[final-report.md](final-report.md)** — the cumulative final report and best-fit recommendation
  across all phases.

---

## 6. Assumptions and open questions

**Assumptions.**

- Self-managed EC2 vLLM/SGLang; SageMaker/EKS only as later productionization.
- Reasoning ON is the baseline for quality (candidate scores assume it).
- Public SWE-bench numbers are indicative only until re-run on the fixed harness.
- 1× `g6e.2xlarge` ≈ $2/hr on-demand, 2× ≈ $4/hr (confirm; use spot/savings plans for long sweeps).

**Open questions to resolve during execution.**

- Exact usable context on 48 GB per precision, given hybrid (Gated DeltaNet) attention — measure the
  real KV footprint rather than assume.
- Which AWQ/INT4 variant matches FP8 quality closely enough to justify its extra KV headroom.
- Whether SGLang's RadixAttention advantage survives the 2-replica HA split (Layer-B dependency).
- Best `qwen3_xml` vs `qwen3_coder` tool-parser behavior under long agentic sessions on our traffic.
- Whether native MTP speculative decoding beats a separate draft model for this workload.
- The exact frontier model(s) and API surface to use as the industry anchor.
