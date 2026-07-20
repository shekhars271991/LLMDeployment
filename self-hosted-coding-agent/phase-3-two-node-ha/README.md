[← Back to index](../PLAN.md)

# Phase 3 — Two-node (HA) and its optimizations

Add a **second `g6e.2xlarge`** so the deployment survives one node failing (real cross-instance
redundancy — the reason to move to 2 nodes). A software router in front exposes a single
OpenAI-compatible `/v1` endpoint (AWS NLB/ALB, or LiteLLM / nginx / the vLLM production-stack router
/ the SGLang router); the software router is preferred because it also carries the Phase 3b routing
experiments. Each node runs one **full-model replica** (data-parallel across nodes).

- **Phase 3a — HA replicas + load balancer (round-robin baseline).** *Goal:* prove HA and measure
  aggregate capacity. Two full replicas behind round-robin. *Measure:* a **failover run** (1 node
  down) confirming degraded-mode capacity still meets SLO, and an **aggregate load/capacity test**
  across the 2 nodes (per-replica → aggregate, see [§3.4 in evaluation.md](../evaluation.md)).
- **Phase 3b — Layer-B serving routing optimizations.** *Goal:* stop naive load balancing from
  wasting the prefix cache. Naive **round-robin breaks prefix caching** (cache-blind → low hit rate →
  wasted prefill). Ladder: round-robin → **session affinity** (sticky) → **prefix-aware**
  (longest-prefix match) → **KV-aware** (global KV index + overlap score vs queue depth). *Tools:*
  vLLM production-stack router (RoundRobin / Session / PrefixAware / KVAware / DisaggregatedPrefill),
  SGLang cache-aware router (`--load-balance-method cache_aware`), NVIDIA Dynamo (global KV block
  index). *Measure:* KV-cache hit rate + concurrent-user capacity at SLO (round-robin vs
  prefix/KV-aware).
- **Phase 3c — Prefill/decode (P/D) disaggregation (scoped experiment).** *Goal:* show the mechanism
  cheaply, then measure a realistic speedup on the right hardware.
  - **The key constraint:** on 2 nodes you get **HA *or* P/D disaggregation, not both.**
    HA-replica mode = 2 independent full replicas (one can die, the other serves). P/D mode
    reconfigures the 2 nodes as 1 prefill + 1 decode, streaming the KV cache between them via a
    vLLM/SGLang connector (NIXL / LMCache / UCX) — two single-points-of-failure **in series** → **not
    HA**. **HA + disaggregation together needs 4 nodes** (2 prefill + 2 decode + router).
  - **Interconnect is the P/D bottleneck, and RDMA is gated by instance size.** KV-cache transfer
    bandwidth tiers, best to worst:

    | Tier | Path | Approx. bandwidth | Available on |
    | --- | --- | --- | --- |
    | Intra-node NVLink/NVSwitch | GPU↔GPU inside one box | ~900 GB/s | `p5.48xlarge`, `p4de.24xlarge` |
    | Cross-node EFA + GPUDirect RDMA | NIC↔NIC, GPU-direct | ~400 GB/s (up to 3200 Gbps EFAv2) | `p5/p5e/p5en/p6.48xlarge`, `p4d/p4de`, some `g6e.48xlarge` |
    | Cross-node TCP | plain networking | ~2.5 GB/s, bursty | `g6e.2xlarge` (no EFA) |

    AWS exposes EFA / GPUDirect RDMA only on the **big multi-GPU** instances. A single-GPU
    `g6e.2xlarge` has **no GPUDirect RDMA**, so two small nodes **cannot** get RDMA.
  - **Mechanism run:** disaggregate across 2× `g6e.2xlarge` over **TCP** (**NOT HA**) — demonstrates
    the P/D *mechanism*, not a best-case speedup.
  - **Realistic-speedup run:** disaggregate prefill/decode across GPUs *inside one NVSwitch box*
    (`p5.48xlarge` / `p4de`) over NVLink — faster than any network RDMA, needs no EFA/placement-group
    setup — **spun up and torn down**, not run 24/7. (Cross-node EFA RDMA on 2× `p5.48xlarge` is the
    production multi-node path but out-of-budget overkill for a 35B.)

**Exit criteria:** recorded **HA capacity numbers** (per-replica / aggregate / failover) and the
**best 2-node serving config** (winning Layer-B routing), plus the P/D mechanism-vs-realistic
findings.

---
**Navigation:** [← Phase 2](../phase-2-single-node-optimizations/README.md) · [Index](../PLAN.md) · [Phase 4 →](../phase-4-routing/README.md)
