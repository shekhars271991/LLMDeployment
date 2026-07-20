[← Back to index](../PLAN.md)

# Phase 4 — Routing (application) and its optimizations

**Layer-A application routing is the centerpiece** — it governs cost & quality (whereas Layer-B in
[Phase 3](../phase-3-two-node-ha/README.md) governs throughput & capacity, not quality). Adopt each rung only if it beats the prior,
measured on the fixed SWE-bench + load harness.

**Decision signals:** task type (autocomplete/edit vs multi-file reasoning vs Q&A), estimated
difficulty, context length, tool-call vs code-gen.

**Strategy ladder:**

1. **Static rules / task tags** — deterministic, ~0 overhead.
2. **Classifier router** — RouteLLM (mf / bert / causal_llm) or an XGBoost model predicting
   P(small model succeeds); RouteLLM reports large cost savings at high quality retention.
3. **Semantic / embedding routing** by inferred intent.
4. **Cascade (cheap-first → verify → escalate)** — the coding-specific centerpiece, because the
   verifier is **executable**: does the patch apply? compile? do repo tests pass? is the tool-call
   schema valid? Escalate to the big model only on failure. This is the pattern that can beat a
   single frontier model on **both** cost and quality, and it maps directly onto SWE-bench (the tests
   *are* the verifier).

**Plus two composable levers:**

- **Method routing** (Phase-2 input, see [phase-2-single-node-optimizations.md](../phase-2-single-node-optimizations/README.md)): speculative-decode method (n-gram / draft / EAGLE-MTP / none)
  per workload — one dimension, not the whole thing.
- **Reasoning-effort routing (high-value lever):** toggle thinking per request via
  `chat_template_kwargs: {enable_thinking: ...}` — reason hard on complex multi-file tasks, off/low
  on trivial edits/autocomplete. Often a bigger cost/latency lever than model choice, and it composes
  with the cascade (reason-off cheap attempt → verify → escalate to reason-on).

**Guardrail:** a **CI eval gate** on a 50–500-case holdout that blocks any routing change dropping
quality below threshold; monitor escalation rate + query-distribution drift.

**Combine:** best-of Layer-A + Layer-B (from [Phase 3](../phase-3-two-node-ha/README.md)) + method routing → report the cost / quality /
capacity **frontier**.

**Evaluation bullets:**

- **Quality:** SWE-bench resolve rate stays within a set threshold of the big-model ceiling.
- **Cost:** blended GPU-time / resolved task; escalation rate; % routed to the small model.
- **Router overhead:** embedding ~20–50 ms; LLM-router ~200 ms+ — confirm net positive.
- **Capacity:** concurrent-user capacity at SLO under the routing policy.

**Exit criteria:** a recorded **routing policy** (adopted Layer-A + Layer-B strategy) plus the
**combined cost/quality/capacity frontier**.

---
**Navigation:** [← Phase 3](../phase-3-two-node-ha/README.md) · [Index](../PLAN.md) · [Final report →](../final-report.md)
