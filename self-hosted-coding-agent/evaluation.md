[← Back to index](PLAN.md)

## 3. Evaluation methodology (cross-cutting — used by every phase)

The shared harness, metrics, and record-keeping that **all phases reuse**. Every phase measures on
this fixed harness so results are comparable across phases and sub-phases.

### 3.1 Headline metric and harness

- **Metric:** SWE-bench **Verified** resolve rate (headline); SWE-bench **Lite** for fast iteration.
- **Harness:** `SWE-agent/mini-swe-agent` — a minimal, model-agnostic scaffold (bash-only, no
  bespoke tool interface) pointed at the self-hosted `/v1` endpoint via litellm
  (`hosted_vllm/<model>` with `api_base`), e.g.:

  ```
  mini-extra swebench --model hosted_vllm/Qwen/Qwen3.6-35B-A3B-FP8 \
      --subset verified --split test
  ```

  Fix the agent scaffold so **only the model/serving config varies** between runs.
- **API-surface control:** SWE-bench scores are harness- and API-sensitive (ChatCompletions vs
  Responses API differences alone can move scores a few points). Standardize on one API surface for
  all self-hosted runs and document it.

### 3.2 Reasoning-aware SLOs and serving metrics

- For a thinking model, raw TTFT is misleading. Track **time-to-first *answer* token** (after
  `</think>`) and total think+answer latency, and use **reasoning-inclusive output lengths** for
  capacity modeling (reasoning inflates output 2–10×, so capacity is wildly overestimated otherwise).
- **Serving metrics recorded alongside quality:** TTFT, decode tok/s, concurrent throughput, peak
  VRAM, tokens/task, and the self-host cost proxy (GPU-hours / resolved task).

### 3.3 Reasoning serving flags (correctness prerequisite)

- **Required flags:** `--reasoning-parser qwen3` + `--enable-auto-tool-choice` +
  `--tool-call-parser qwen3_xml`. Prefer `qwen3_xml` over `qwen3_coder`: the XML parser survives long
  agentic sessions, whereas `qwen3_coder` can drop tool calls emitted inside `<think>`. Pin
  **vLLM ≥ 0.24** for the reasoning + tool streaming parser.
- **`preserve_thinking: true`** retains reasoning across turns → better agent consistency and lets
  prefix caching reuse prior turns instead of re-thinking.

### 3.4 Load & capacity benchmark (throughput + concurrent users)

A **separate** concurrent load test, distinct from the concurrency-1 latency runs:

- **Define SLOs first:** p95 TTFT and p95 inter-token latency (ITL/TPOT) thresholds + a minimum
  interactive per-user decode rate (e.g. ≥ ~15 tok/s). Capacity is defined against these.
- **Sweep** concurrency (1, 2, 4, 8, 16, 32, 64, …) and/or request rate (Poisson λ) with a tool like
  `vllm bench serve`, SGLang `bench_serving`, or guidellm, using coding-representative input/output
  length distributions (ideally sampled from SWE-bench trajectories).
- **Record** throughput (output tok/s, req/s), TTFT/ITL/e2e percentiles, and goodput (share meeting
  SLO).
- **Capacity** = the SLO "knee" (max in-flight concurrency where p95 still meets SLO).
- **Convert to concurrent users** via Little's Law (L = λ × W) + a per-user duty cycle (request rate
  × server time per request); document the workload assumptions explicitly.

### 3.5 Industry reference baselines (position against what you'd otherwise buy)

- **Apples-to-apples anchor:** run ≥1 frontier commercial model (Claude Opus/Code, GPT-5.x Codex,
  Gemini 3.x) through the **identical** mini-swe-agent harness via API — the only fair cross-system
  comparison, since SWE-bench scores are harness-sensitive.
- **Context (public numbers, 2026):** Claude Opus 4.8 ~88.6%, Claude Code ~72–78%, GPT-5.x Codex
  ~76–83%, Cursor ~67–68%, Devin ~58–61%, OpenHands ~52%, open-weight-backed agents ~32–38%.
  The 35B target (~73%) sits in **Claude-Code / Cursor territory** and beats existing open
  self-hosted agents outright.
- **Cost comparison:** self-host GPU-$ / resolved task vs commercial API $/task (Claude Code
  ~$1.5–3, Cursor ~$0.4–0.9, Devin ~$3–6, OpenHands ~$0.10). Value proposition = near-commercial
  quality at a fraction of $/task, fully self-hosted.

### 3.6 Validity controls

Fixed configs per run, cross-checked server launch logs, medians across repeats — mirroring the
speculative-decoding project.

### 3.7 Per-phase record template (used to close every phase / sub-phase)

Each phase and sub-phase records the following so results accumulate consistently:

- **Config:** model + precision + engine + serving flags + routing/topology (the exact changed lever).
- **Quality:** SWE-bench resolve rate (Lite for iteration, Verified for headline) and delta vs the
  current baseline.
- **Serving metrics:** TTFT / time-to-first-answer-token, decode tok/s, peak VRAM, tokens/task.
- **Capacity:** SLO-knee concurrency and estimated concurrent users at SLO (per §3.4).
- **Cost proxy:** GPU-hours / resolved task (and blended $/resolved-task).
- **Notes / decision:** adopt or reject vs baseline, and why (the reason the next phase inherits it).
