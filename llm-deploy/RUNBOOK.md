# RUNBOOK — Qwen3-32B AWQ baseline on vLLM (AWS)

Read top-to-bottom. Run one step at a time; record notes in **Results** at the bottom.

**You run everything.** These scripts are small helpers — inspect each before executing.

## Automatic raw records

Shell scripts use `common/record.sh`, and Python scripts use
`common/recording.py`, to keep console output and save the same stdout/stderr
to timestamped raw logs automatically:

- Mac/AWS scripts: `records/aws/<timestamp>_<script>.log`
- EC2/remote scripts: `remote/records/<timestamp>_<script>.log`
- Benchmark data: `remote/results/bench_<timestamp>.json`
- Summaries: `remote/results/summary_<timestamp>.txt`

You do not need to copy console output manually. These generated records are
ignored by Git because they may contain instance IDs, IPs, and verbose logs.

## Local configuration safety

`aws/config.env` and `remote/config.env` are local files ignored by Git.
Tracked `.example` files contain placeholders only. On a fresh clone:

```bash
cp aws/config.env.example aws/config.env
cp remote/config.env.example remote/config.env
```

Never put AWS credentials, PEM contents, account IDs, VPC/security-group IDs,
instance IDs, or public IPs in tracked files. `aws/instance.env`, raw records,
PEM files, benchmark results, and Terraform state are also ignored.

---

## Hardware decision (pick before launch)

| Instance | GPUs | VRAM | TP | On-demand (us-east-1, ~Jul 2026) | Use when |
|----------|------|------|-----|----------------------------------|----------|
| `g6e.xlarge` | 1× L40S | 48 GB | 1 | ~$1.86/hr | **Start here** — simplest baseline |
| `g6e.12xlarge` | 4× L40S | 192 GB | 2 (`CUDA_VISIBLE_DEVICES=0,1`) | TP=2 + future optimizations |
| `g5.12xlarge` | 4× A10G | 96 GB | 2 (budget) | Cheaper TP=2; NCCL PCIe quirks |

Set `INSTANCE_TYPE` in `aws/config.env`. For TP=2 on a 4-GPU box, also set in `remote/config.env`:
```bash
export TP=2
export CUDA_VISIBLE_DEVICES="0,1"
export NCCL_P2P_DISABLE="1"          # if serve hangs at 100% GPU
export DISABLE_CUSTOM_ALL_REDUCE="1"   # optional, same hang scenario
```

---

## Step 0 — One-time AWS console setup (manual)

Do these in AWS Console before scripts:

1. **Region** — `us-east-1`, `us-east-2`, or `us-west-2`. Put in `aws/config.env`.
2. **Service quota** — request **Running On-Demand G and VT instances** (vCPUs). `g6e.xlarge` = 4 vCPUs; `g6e.12xlarge` = 48.
3. **Key pair** — create/download `.pem`. Set `KEY_NAME` and `SSH_KEY_PATH` in `aws/config.env`.
4. **Security group** — allow **SSH (22) from your IP only**. Do **not** open port 8000 (use SSH tunnel).
5. **AMI** — find **Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)** for your region. Set `AMI_ID` in `aws/config.env`.

**Record:** region, instance type chosen, quota approved? (yes/no)

---

## Step 1 — `aws/check_quota.sh`

**What:** Prints your G/VT vCPU quota vs what `INSTANCE_TYPE` needs.

**Where:** Mac, from `llm-deploy/`

```bash
bash aws/check_quota.sh
```

**Good result:** "OK: quota appears sufficient"

**Record:** current vCPU limit = ___

---

## Step 2 — `aws/launch_instance.sh`

**What:** Launches EC2 from `aws/config.env`; writes `aws/instance.env` with IP.

**Prerequisite:** Step 0 done; `AMI_ID`, `KEY_NAME`, `SG_ID` filled in.

```bash
bash aws/launch_instance.sh
```

**Good result:** `PUBLIC_IP=...` printed; instance state `running`.

**Record:** instance id = ___ | public IP = ___

---

## Step 3 — Copy `remote/` and the recorder to the box

**What:** Copy benchmark scripts and their shared recorder to the GPU instance.

**Prerequisite:** Step 2; replace IP and key path.

```bash
scp -r -i ~/.ssh/<key>.pem remote common ubuntu@<PUBLIC_IP>:~/
```

**Good result:** `~/remote/` and `~/common/` exist on the box.

---

## Step 4 — `aws/ssh_tunnel.sh`

**What:** SSH session with `localhost:8000` → remote vLLM port.

**Where:** Mac (keep this terminal open while testing from Mac; remote scripts use `127.0.0.1` on the box itself).

```bash
bash aws/ssh_tunnel.sh
```

**Good result:** Logged into Ubuntu shell on the GPU box.

Open a **second** SSH session for running remote scripts (no tunnel needed on that one):
```bash
ssh -i ~/.ssh/<key>.pem ubuntu@<PUBLIC_IP>
```

---

## Step 5 — `remote/setup_venv.sh`

**What:** Installs `python3-venv` if missing, creates a fresh venv, installs
`vllm>=0.8.5`, and prints the vLLM version and `nvidia-smi`.

**Where:** EC2 box, in `~/remote/`

```bash
cd ~/remote
bash setup_venv.sh
```

**Good result:** GPU(s) listed; vLLM version ≥ 0.8.5.

**Record:** GPU model = ___ | vLLM version = ___

---

## Step 6 — `remote/download_weights.sh`

**What:** Pre-downloads `Qwen/Qwen3-32B-AWQ` (~20 GB) to `HF_HOME`.

```bash
cd ~/remote
source ~/vllm-venv/bin/activate
bash download_weights.sh
```

Optional: `hf auth login` if rate-limited.

**Good result:** "Download complete" under `~/hf`.

**Record:** download time = ___ min

---

## Step 7 — `remote/serve.sh`

**What:** Starts vLLM OpenAI API server — baseline config, no extra optimizations.

**Prerequisite:** Edit `remote/config.env` if using TP=2.

Run in a **dedicated terminal** (blocks):

```bash
cd ~/remote
source ~/vllm-venv/bin/activate
bash serve.sh
```

**Good result:** `Application startup complete` / Uvicorn on port 8000.

**Record:** startup time = ___ min | TP = ___ | MAXLEN = ___

---

## Step 8 — `remote/smoke_test.sh`

**What:** Lists models + one short chat completion.

**Prerequisite:** `serve.sh` running in another terminal.

```bash
cd ~/remote
bash smoke_test.sh
```

**Good result:** JSON with model id and a text reply.

---

## Step 9 — `remote/bench.py`

**What:** Runs fixed prompts at context lengths 256 / 2048 / 8000 (sequential, c=1). Logs TTFT, decode tok/s, peak VRAM per request.

```bash
cd ~/remote
source ~/vllm-venv/bin/activate
python bench.py
```

**Good result:** `results/bench_<timestamp>.json` created; no errors per prompt.

Optional cross-check (random prompts, vLLM built-in):
```bash
vllm bench serve --base-url http://127.0.0.1:8000 --model qwen3-32b-awq \
  --dataset-name random --random-input-len 2048 --random-output-len 256 \
  --num-prompts 10 --max-concurrency 1
```

**Record:** bench output file = ___

---

## Step 10 — `remote/summarize.py`

**What:** Aggregates latest `results/bench_*.json` into a table (TTFT + decode tok/s + peak VRAM by context length).

```bash
cd ~/remote
source ~/vllm-venv/bin/activate
python summarize.py
```

**Good result:** Table printed; `results/summary_<timestamp>.txt` written.

Paste the summary into **Results** below.

---

## Step 11 — `aws/terminate.sh`

**What:** Terminates the EC2 instance when finished (stops billing for compute).

**Where:** Mac, from `llm-deploy/`

```bash
bash aws/terminate.sh
```

Type `yes` to confirm.

---

## Script reference (short)

| Script | One line |
|--------|----------|
| `common/record.sh` | Tee shell-script stdout/stderr into timestamped raw logs |
| `common/recording.py` | Tee Python stdout/stderr into timestamped raw logs |
| `aws/config.env` | Shared AWS settings — edit before launch |
| `aws/check_quota.sh` | Check G/VT vCPU quota |
| `aws/launch_instance.sh` | Launch GPU EC2 instance |
| `aws/ssh_tunnel.sh` | SSH + port-forward 8000 |
| `aws/terminate.sh` | Tear down instance |
| `remote/config.env` | Model, TP, MAXLEN, ports — edit before serve/bench |
| `remote/setup_venv.sh` | Install vLLM in fresh venv |
| `remote/download_weights.sh` | HF download Qwen3-32B-AWQ |
| `remote/serve.sh` | Start vLLM server (baseline) |
| `remote/smoke_test.sh` | Quick API sanity check |
| `remote/prompts.json` | Fixed 8-prompt benchmark set |
| `remote/bench.py` | Measure TTFT / tok/s / VRAM |
| `remote/summarize.py` | Aggregate bench JSON to table |

---

## Results (fill in after Step 10)

**Run metadata**

| Field | Value |
|-------|-------|
| Date | |
| Instance type | |
| GPU | |
| vLLM version | |
| TP | |
| MAXLEN | |
| GPU_MEM_UTIL | |

**Metrics by context length** (from `summarize.py`)

| ctx_tokens | n | TTFT mean (ms) | TTFT median (ms) | decode mean (tok/s) | decode median (tok/s) | VRAM max (MiB) |
|------------|---|----------------|------------------|---------------------|----------------------|----------------|
| 256 | | | | | | |
| 2048 | | | | | | |
| 8000 | | | | | | |

**Notes / observations**

- 
- 

---

## Troubleshooting

- **Quota insufficient** — Service Quotas → EC2 → increase G/VT on-demand vCPUs.
- **Serve hangs at 100% GPU (TP=2 on A10G)** — set `NCCL_P2P_DISABLE=1` and `DISABLE_CUSTOM_ALL_REDUCE=1` in `remote/config.env`.
- **OOM at startup** — lower `MAXLEN` (e.g. 8192) or use `g6e.xlarge` (48 GB).
- **Extra "thinking" tokens** — bench already sends `enable_thinking: false`; smoke_test does not (fine for sanity check only).
