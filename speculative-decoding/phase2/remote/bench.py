#!/usr/bin/env python3
"""
bench.py — Phase 2 benchmark with cohorts, deterministic fixed-length outputs,
and speculative-decoding metrics from /metrics.

Prerequisite: a Phase 2 serve script running; speculative_prompts.json present.
Run on remote box:
  export BENCH_RUN_LABEL=control   # or ngram-4, draft, etc.
  ./bench.py

Writes: results/bench_<label>_<timestamp>.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
COMMON_DIR = SCRIPT_DIR.parent.parent / "common"

sys.path.insert(0, str(COMMON_DIR))
from recording import start_recording  # noqa: E402

start_recording("bench", SCRIPT_DIR / "records")

SPEC_METRIC_KEYS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def load_config_env() -> dict[str, str]:
    cfg: dict[str, str] = {}
    env_path = SCRIPT_DIR / "config.env"
    if not env_path.exists():
        return cfg
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        val = os.path.expandvars(val)
        cfg[key.strip()] = val
        os.environ.setdefault(key.strip(), val)
    return cfg


def api_root_from_base(api_base: str) -> str:
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def tokenize_chat(api_base: str, model: str, content: str) -> int:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{api_root_from_base(api_base)}/tokenize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(json.load(response)["count"])


FINE_FILLER_TOKEN = " padding"


def build_varied_filler(units: int, seed: str) -> str:
    """Unique short segments so padding does not create artificial n-gram reuse."""
    return "".join(f" seg-{seed}-{i % 113} " for i in range(max(0, units)))


def pad_prompt(
    api_base: str,
    model: str,
    prompt_id: str,
    prompt: str,
    target_input_tokens: int,
) -> tuple[str, int]:
    base = (
        f"Benchmark case: {prompt_id}-{target_input_tokens}\n\n"
        f"{prompt}\n\n"
        "Ignore the following varied padding:"
    )
    base_tokens = tokenize_chat(api_base, model, base)
    if base_tokens > target_input_tokens:
        raise ValueError(
            f"{prompt_id} requires {base_tokens} tokens before padding, "
            f"which exceeds target {target_input_tokens}. "
            f"Increase BENCH_CONTEXT_LENGTHS minimum or shorten the prompt."
        )
    if base_tokens == target_input_tokens:
        return base, base_tokens

    seed = f"{prompt_id}-{target_input_tokens}"

    def assemble(segments: int, fine_repeats: int) -> str:
        text = base + build_varied_filler(segments, seed)
        if fine_repeats > 0:
            text += FINE_FILLER_TOKEN * fine_repeats
        return text

    def token_count(segments: int, fine_repeats: int) -> int:
        return tokenize_chat(api_base, model, assemble(segments, fine_repeats))

    deficit = target_input_tokens - base_tokens
    lo_seg, hi_seg = 0, max(deficit * 4, 64)
    while token_count(hi_seg, 0) < target_input_tokens:
        hi_seg *= 2
        if hi_seg > 200_000:
            raise RuntimeError(
                f"Could not pad {prompt_id} to exactly {target_input_tokens} tokens"
            )

    while lo_seg < hi_seg:
        mid = (lo_seg + hi_seg + 1) // 2
        if token_count(mid, 0) <= target_input_tokens:
            lo_seg = mid
        else:
            hi_seg = mid - 1

    for segments in range(max(0, lo_seg - 1), lo_seg + 4):
        at_segments = token_count(segments, 0)
        if at_segments > target_input_tokens:
            continue
        fine_hi = max(target_input_tokens - at_segments + 32, 64)
        fine_lo = 0
        while fine_lo <= fine_hi:
            mid = (fine_lo + fine_hi) // 2
            actual = token_count(segments, mid)
            if actual == target_input_tokens:
                return assemble(segments, mid), actual
            if actual < target_input_tokens:
                fine_lo = mid + 1
            else:
                fine_hi = mid - 1

    raise RuntimeError(
        f"Could not pad {prompt_id} to exactly {target_input_tokens} tokens "
        f"(base={base_tokens}, segments={lo_seg}, at_segments={token_count(lo_seg, 0)})"
    )


def parse_prometheus_counters(text: str, names: tuple[str, ...]) -> dict[str, float]:
    values = {name: 0.0 for name in names}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in names:
            if not line.startswith(name):
                continue
            match = re.search(r"\}\s+([0-9.eE+-]+)$", line)
            if match:
                values[name] += float(match.group(1))
    return values


def fetch_spec_metrics(api_base: str) -> dict[str, float]:
    url = f"{api_root_from_base(api_base)}/metrics"
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    counters = parse_prometheus_counters(text, SPEC_METRIC_KEYS)
    drafts = counters["vllm:spec_decode_num_drafts"]
    draft_tokens = counters["vllm:spec_decode_num_draft_tokens_total"]
    accepted = counters["vllm:spec_decode_num_accepted_tokens_total"]
    acceptance_rate = (accepted / draft_tokens) if draft_tokens > 0 else None
    mean_accepted_length = (1.0 + accepted / drafts) if drafts > 0 else None
    return {
        **counters,
        "acceptance_rate": acceptance_rate,
        "mean_accepted_length": mean_accepted_length,
    }


class VramSampler:
    def __init__(self) -> None:
        self._peak_mib: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                for line in out.strip().splitlines():
                    mib = int(line.strip())
                    self._peak_mib = max(self._peak_mib, mib)
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                pass
            self._stop.wait(0.25)

    def start(self) -> None:
        self._peak_mib = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self._peak_mib


@dataclass
class RequestResult:
    prompt_id: str
    cohort: str
    category: str
    target_context_tokens: int
    ttft_ms: float
    decode_tok_s: float | None
    total_elapsed_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    peak_vram_mib: int
    error: str | None = None


@dataclass
class BenchRun:
    timestamp: str
    run_label: str
    model: str
    api_base: str
    config: dict[str, Any]
    context_lengths: list[int]
    metrics_before: dict[str, float | None]
    metrics_after: dict[str, float | None]
    metrics_delta: dict[str, float | None]
    results: list[RequestResult] = field(default_factory=list)


def stream_chat(
    api_base: str,
    model: str,
    content: str,
    max_tokens: int,
    temperature: float,
) -> tuple[float | None, float, dict[str, int | None], str | None]:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0 else 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"ignore_eos": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft: float | None = None
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk == "[DONE]":
                    break
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                if event.get("usage"):
                    u = event["usage"]
                    usage["prompt_tokens"] = u.get("prompt_tokens")
                    usage["completion_tokens"] = u.get("completion_tokens")
                    usage["total_tokens"] = u.get("total_tokens")

                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return None, (time.perf_counter() - t0) * 1000, usage, f"HTTP {e.code}: {body[:500]}"
    except Exception as exc:  # noqa: BLE001
        return None, (time.perf_counter() - t0) * 1000, usage, str(exc)

    total_ms = (time.perf_counter() - t0) * 1000
    return ttft, total_ms, usage, None


def metrics_delta(
    before: dict[str, float | None],
    after: dict[str, float | None],
) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for key in set(before) | set(after):
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            delta[key] = None
        else:
            delta[key] = a - b
    drafts = delta.get("vllm:spec_decode_num_drafts")
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens_total")
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total")
    if draft_tokens and draft_tokens > 0 and accepted is not None:
        delta["acceptance_rate"] = accepted / draft_tokens
    else:
        delta["acceptance_rate"] = None
    if drafts and drafts > 0 and accepted is not None:
        delta["mean_accepted_length"] = 1.0 + accepted / drafts
    else:
        delta["mean_accepted_length"] = None
    return delta


def main() -> int:
    load_config_env()

    run_label = os.environ.get("BENCH_RUN_LABEL", "control")
    model = os.environ.get("SERVED_MODEL_NAME", "qwen3-32b-awq")
    api_base = os.environ.get("BENCH_API_BASE", "http://127.0.0.1:8000/v1")
    max_tokens = int(os.environ.get("BENCH_MAX_TOKENS", "256"))
    temperature = float(os.environ.get("BENCH_TEMPERATURE", "0"))
    prompts_file = SCRIPT_DIR / os.environ.get(
        "BENCH_PROMPTS_FILE", "speculative_prompts.json"
    )
    context_lengths = [
        int(x.strip())
        for x in os.environ.get("BENCH_CONTEXT_LENGTHS", "256,2048,8000").split(",")
        if x.strip()
    ]

    if "${" in api_base:
        print(f"ERROR: BENCH_API_BASE contains an unexpanded variable: {api_base}", file=sys.stderr)
        return 1

    try:
        with urllib.request.urlopen(f"{api_base.rstrip('/')}/models", timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: vLLM API preflight failed at {api_base}: {exc}", file=sys.stderr)
        return 1

    if not prompts_file.exists():
        print(f"ERROR: {prompts_file} not found", file=sys.stderr)
        return 1

    prompts = json.loads(prompts_file.read_text())["prompts"]
    metrics_before = fetch_spec_metrics(api_base)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^a-zA-Z0-9._-]+", "-", run_label)
    out_path = RESULTS_DIR / f"bench_{safe_label}_{ts}.json"

    run = BenchRun(
        timestamp=ts,
        run_label=run_label,
        model=model,
        api_base=api_base,
        config={
            "TP": os.environ.get("TP"),
            "MAXLEN": os.environ.get("MAXLEN"),
            "GPU_MEM_UTIL": os.environ.get("GPU_MEM_UTIL"),
            "BENCH_MAX_TOKENS": max_tokens,
            "BENCH_TEMPERATURE": temperature,
            "BENCH_CONTEXT_LENGTHS": context_lengths,
            "NGRAM_NUM_SPEC_TOKENS": os.environ.get("NGRAM_NUM_SPEC_TOKENS"),
            "DRAFT_MODEL_ID": os.environ.get("DRAFT_MODEL_ID"),
            "DRAFT_NUM_SPEC_TOKENS": os.environ.get("DRAFT_NUM_SPEC_TOKENS"),
        },
        context_lengths=context_lengths,
        metrics_before=metrics_before,
        metrics_after={},
        metrics_delta={},
    )

    print(f"Benchmark: label={run_label} model={model} contexts={context_lengths}")
    print(f"Prompts: {len(prompts)} file={prompts_file.name}")
    print(f"API: {api_base}")
    print(f"Generation: temperature={temperature} max_tokens={max_tokens} ignore_eos=true")
    print("Concurrency: 1 (sequential)\n")

    for ctx in context_lengths:
        print(f"--- input length {ctx} tokens ---")
        for prompt in prompts:
            pid = prompt["id"]
            cohort = prompt.get("cohort", "unknown")
            try:
                padded, prepared_tokens = pad_prompt(
                    api_base, model, pid, prompt["text"], ctx
                )
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR preparing {pid} @ ctx={ctx}: {exc}", file=sys.stderr)
                return 1

            print(f"  {pid} ({cohort}) @ ctx={ctx} ... ", end="", flush=True)

            sampler = VramSampler()
            sampler.start()
            ttft_ms, total_ms, usage, err = stream_chat(
                api_base, model, padded, max_tokens, temperature
            )
            peak_mib = sampler.stop()

            decode_tok_s: float | None = None
            comp = usage.get("completion_tokens")
            if ttft_ms is not None and comp and comp > 1:
                decode_sec = max(1e-9, (total_ms - ttft_ms) / 1000)
                decode_tok_s = (comp - 1) / decode_sec

            result = RequestResult(
                prompt_id=pid,
                cohort=cohort,
                category=prompt.get("category", ""),
                target_context_tokens=ctx,
                ttft_ms=ttft_ms or -1.0,
                decode_tok_s=decode_tok_s,
                total_elapsed_ms=total_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                peak_vram_mib=peak_mib,
                error=err,
            )
            run.results.append(result)

            if err:
                print(f"ERROR: {err}")
            elif ttft_ms is not None:
                dts = f"{decode_tok_s:.1f}" if decode_tok_s else "n/a"
                actual_tokens = usage.get("prompt_tokens")
                token_note = (
                    str(actual_tokens)
                    if actual_tokens is not None
                    else f"{prepared_tokens} prepared"
                )
                comp_note = usage.get("completion_tokens")
                print(
                    f"input={token_note} out={comp_note} "
                    f"TTFT={ttft_ms:.0f}ms decode={dts} tok/s VRAM={peak_mib}MiB"
                )
            else:
                print("no tokens received")

            time.sleep(0.5)

    metrics_after = fetch_spec_metrics(api_base)
    run.metrics_after = metrics_after
    run.metrics_delta = metrics_delta(metrics_before, metrics_after)

    out_path.write_text(json.dumps({"run": asdict(run)}, indent=2))
    print(f"\nWrote {out_path}")
    if run.metrics_delta.get("vllm:spec_decode_num_draft_tokens_total"):
        print(
            "Speculative delta: "
            f"draft_tokens={run.metrics_delta['vllm:spec_decode_num_draft_tokens_total']:.0f} "
            f"accepted={run.metrics_delta.get('vllm:spec_decode_num_accepted_tokens_total', 0):.0f} "
            f"acceptance_rate={run.metrics_delta.get('acceptance_rate')}"
        )
    print("Next: ./compare_speculative.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
