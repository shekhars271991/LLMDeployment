# Phase 2 — Speculative Decoding Runbook

Manual, step-by-step guide for n-gram and draft-model experiments on the L40S.

**Do not modify `phase1/`.** All Phase 2 work lives here.

**Do not modify `phase1/`.** All Phase 2 work lives here.

**First-time checklist**

1. Mac: `source aws/load_env.sh`
2. Mac: copy files to EC2 (**Step 1**)
3. EC2: `cd ~/phase2/remote` then setup (**Step 2**)
4. EC2: run experiments (**Step 3+**)

> Scripts live in **`~/phase2/remote/`**, not `~/phase2/`.

---

## Step 0 — Load environment on Mac (always first)

Run at the start of **every Mac session** before scp, ssh, or tunnel.

Infra settings: `aws/config.env`. Live instance details: `aws/instance.env` (written by `launch_instance.sh`).

```bash
cd /path/to/LLMDeployment/speculative-decoding
source aws/load_env.sh

echo "Instance: ${INSTANCE_ID} @ ${INSTANCE_PUBLIC_IP}"
echo "Type: ${INSTANCE_TYPE}  Region: ${AWS_REGION}"
echo "SSH: ${SSH_USER}@${INSTANCE_PUBLIC_IP}"
echo "Key: ${SSH_KEY_PATH}"
```

If `instance.env` is missing:

```bash
bash aws/check_quota.sh
bash aws/launch_instance.sh
source aws/load_env.sh
```

Optional tunnel:

```bash
bash aws/ssh_tunnel.sh
```

---

## Step 1 — Copy Phase 2 files to EC2 (Mac)

Run from `speculative-decoding/` **after Step 0**. Re-run when scripts change.

```bash
source aws/load_env.sh

ssh -i "${SSH_KEY_PATH}" "${SSH_USER}@${INSTANCE_PUBLIC_IP}" "mkdir -p ~/phase2"

scp -r -i "${SSH_KEY_PATH}" \
  phase2/remote \
  "${SSH_USER}@${INSTANCE_PUBLIC_IP}:~/phase2/"

scp -r -i "${SSH_KEY_PATH}" \
  common \
  "${SSH_USER}@${INSTANCE_PUBLIC_IP}:~/"
```

This creates **`~/phase2/remote/`** on EC2 and **`~/common/`** for logging. Your old **`~/remote/`** (Phase 1) is left untouched.

Verify on EC2:

```bash
ls ~/phase2/remote/load_env.sh ~/phase2/remote/serve_ngram_speculative.sh
```

---

## Step 2 — EC2 setup (once, then load env every session)

### First-time setup on EC2

Prerequisites: Phase 1 already created `~/vllm-venv` and downloaded Qwen3-32B-AWQ on this box.

```bash
cd ~/phase2/remote
chmod +x *.sh *.py
cp config.env.example config.env
source load_env.sh

echo "Model: ${MODEL_ID}  Port: ${PORT}  Label: ${BENCH_RUN_LABEL}"
echo "Venv: ${VENV_DIR}  HF cache: ${HF_HOME}"
```

### Every new EC2 SSH session

```bash
cd ~/phase2/remote
source load_env.sh
```

**Fields to verify in `config.env`**

| Variable | Typical value | Notes |
|----------|---------------|-------|
| `VENV_DIR` | `${HOME}/vllm-venv` | From Phase 1 setup |
| `HF_HOME` | `${HOME}/hf` | Target + draft weights cache here |
| `GPU_MEM_UTIL` | `0.90` | Lower to `0.85` if draft serve OOMs |
| `BENCH_RUN_LABEL` | `control` | Override per run: `ngram-2`, `ngram-4`, `ngram-8`, `draft` |
| `NGRAM_NUM_SPEC_TOKENS` | `4` | Set to `2`, `4`, or `8` before each n-gram sweep |
| `DRAFT_MODEL_ID` | `Qwen/Qwen3-0.6B` | Used by draft serve + download script |

Optional: `export HF_TOKEN="hf_..."` in the shell before downloads.

Per-run label without editing the file: `export BENCH_RUN_LABEL=ngram-4` then `./bench.py`.

Serve/bench scripts also read `config.env` internally, but **load it in your shell first** so `curl` and exports use the same values.

---

## Step 3 — Run experiments

## Script reference

| Script | Purpose |
|--------|---------|
| `load_env.sh` | Source `config.env` into the current shell (Step 0 on EC2) |
| `serve.sh` | Control server (no speculation) |
| `serve_ngram_speculative.sh` | N-gram speculation (`NGRAM_NUM_SPEC_TOKENS`) |
| `serve_draft_speculative.sh` | Draft model (`Qwen/Qwen3-0.6B`) |
| `download_draft_weights.sh` | Download draft model to `HF_HOME` |
| `bench.py` | Benchmark with cohorts + spec metrics |
| `compare_speculative.py` | Control vs speculative comparison |
| `smoke_test.sh` | Quick API sanity check |
| `speculative_prompts.json` | High / medium / low reuse prompt cohorts |

Shared recorder: `~/common/record.sh` and `~/common/recording.py`

Mac infra loader: `aws/load_env.sh` (sources `aws/config.env` + `aws/instance.env`)

---

## Run order

Each block below assumes **`cd ~/phase2/remote && source load_env.sh`** in that terminal.

### 3a. Draft weights (once)

```bash
cd ~/phase2/remote
source load_env.sh
source "${VENV_DIR}/bin/activate"
./download_draft_weights.sh
```

### 3b. Control benchmark (3 runs)

Terminal A — start control server:

```bash
cd ~/phase2/remote
source load_env.sh
./serve.sh
```

Terminal B — benchmark three times:

```bash
cd ~/phase2/remote
source load_env.sh
source "${VENV_DIR}/bin/activate"
export BENCH_RUN_LABEL=control
./bench.py   # repeat 3 times
```

Stop server with `Ctrl+C` in Terminal A when done.

### 3c. N-gram sweep (2, 4, 8 tokens × 3 runs each)

Edit `~/phase2/remote/config.env`, set `NGRAM_NUM_SPEC_TOKENS` to `2`, then `4`, then `8`. Reload:

```bash
source load_env.sh
```

For each depth, restart server and run bench 3 times:

```bash
# Terminal A
cd ~/phase2/remote && source load_env.sh
./serve_ngram_speculative.sh

# Terminal B
cd ~/phase2/remote && source load_env.sh
source "${VENV_DIR}/bin/activate"
export BENCH_RUN_LABEL=ngram-2   # or ngram-4, ngram-8
./bench.py
```

Check `/metrics` shows increasing `spec_decode_*` counters after the first run.

### 3d. Draft-model benchmark (3 runs)

```bash
# Terminal A
cd ~/phase2/remote && source load_env.sh
./serve_draft_speculative.sh
```

If OOM at startup, set `GPU_MEM_UTIL=0.85` in `config.env`, run `source load_env.sh`, then rerun **control** and **draft** at 0.85.

```bash
# Terminal B
cd ~/phase2/remote && source load_env.sh
source "${VENV_DIR}/bin/activate"
export BENCH_RUN_LABEL=draft
./bench.py   # 3 times
```

### 3e. Compare and copy results home

```bash
cd ~/phase2/remote
source load_env.sh
./compare_speculative.py
```

**On Mac** — after `source aws/load_env.sh`:

```bash
scp -r -i "${SSH_KEY_PATH}" \
  "${SSH_USER}@${INSTANCE_PUBLIC_IP}:~/phase2/remote/{results,records}" \
  phase2/remote/

scp -i "${SSH_KEY_PATH}" \
  "${SSH_USER}@${INSTANCE_PUBLIC_IP}:~/phase2/phase2_speculative_results.txt" \
  phase2/ 2>/dev/null || true
```

(`compare_speculative.py` writes the report to `phase2/phase2_speculative_results.txt`.)

---

## What to expect

| Cohort | N-gram expectation |
|--------|-------------------|
| `high_reuse` | Higher acceptance, best decode speedup |
| `medium_reuse` | Moderate acceptance |
| `low_reuse` | Low acceptance; may be slower than control |

Draft model may beat n-gram on low-reuse tasks if acceptance is higher, at the cost of extra VRAM.

## Fixed benchmark settings

- Prefix caching: **disabled** on all servers
- Input lengths: 256, 2048, 8000 (exact via `/tokenize`)
- Output: **256 tokens**, `temperature=0`, `ignore_eos=true`
- Concurrency: 1
- Prompt file: `speculative_prompts.json` (9 prompts, 3 per cohort)

## Results location

- Raw logs: `phase2/remote/records/`
- JSON benchmarks: `phase2/remote/results/bench_<label>_<timestamp>.json`
- Comparison: `phase2/remote/results/compare_*.txt`
- Final report: `phase2/phase2_speculative_results.txt`

## Troubleshooting

- **No spec_decode metrics** — control mode is expected; speculative servers must be running before bench.
- **OOM with draft model** — lower `GPU_MEM_UTIL` to 0.85 and rerun control + draft at the same setting.
- **Completion tokens ≠ 256** — check server logs; `ignore_eos` requires recent vLLM (0.25.x).
- **Empty `${INSTANCE_PUBLIC_IP}`** — run `source aws/load_env.sh` on Mac; relaunch if instance was terminated.

Reference: [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
