# LLM Deployment Experiments

Hands-on deployment and performance experiments for Qwen3-32B AWQ with vLLM.

See [RESULTS.md](RESULTS.md) for combined Phase 1 and Phase 2 findings.

## Structure

- `aws/` — shared EC2 lifecycle scripts; `instance.env` stores live infra after launch
- `aws/load_env.sh` — source on Mac before scp/ssh (loads `config.env` + `instance.env`)
- `common/` — shared output-recording utilities
- `phase1/` — completed, immutable single-L40S baseline
- `phase2/` — speculative-decoding experiments ([RUNBOOK](phase2/RUNBOOK.md))
- `records/` — AWS script logs

Each phase contains its own scripts, configuration templates, results, and learnings. Do not modify completed phase folders.
