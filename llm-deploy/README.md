# LLM Deployment Experiments

Hands-on deployment and performance experiments for Qwen3-32B AWQ with vLLM.

## Structure

- `aws/` — shared EC2 lifecycle scripts
- `common/` — shared output-recording utilities
- `phase1/` — completed, immutable single-L40S baseline
- `phase2/` — speculative-decoding experiments
- `records/` — AWS script logs

Each phase contains its own scripts, configuration templates, results, and learnings. Do not modify completed phase folders.
