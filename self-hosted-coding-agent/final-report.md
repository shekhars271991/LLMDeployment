[← Back to index](PLAN.md)

# Final report

Cumulative across all phases:

1. **Winning configuration** (model + precision + engine + optimization set + routing policy) and why.
2. **Quality vs cost frontier**, positioned against industry coding agents (Claude Code / Cursor /
   Devin / OpenHands / open-weight-backed), anchored by a frontier model on the **same harness**.
3. **Capacity at SLO:** concurrent users, per-replica, aggregate, and failover (degraded mode).
4. **HA and P/D-disaggregation findings** (mechanism vs realistic-speedup, RDMA limits).
5. **Adopted routing policy:** the Layer-A + Layer-B strategy, escalation rate, savings.
6. **Scale-up recommendation** (which model + topology at the next GPU tier, per [§2.4 in reference.md](reference.md)).
7. Open questions and follow-ups.
