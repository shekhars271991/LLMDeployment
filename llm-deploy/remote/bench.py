#!/usr/bin/env python3
"""
bench.py — Stream chat completions; log TTFT, decode tok/s, VRAM per prompt/context length.

Prerequisite: serve.sh running; prompts.json present.
Run on remote box:
  source ~/vllm-venv/bin/activate
  python bench.py

Writes: results/bench_<timestamp>.json
"""

from __future__ import annotations

import json
import os
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
PROMPTS_FILE = SCRIPT_DIR / "prompts.json"

# Approximate chars per token for padding (good enough for baseline sweeps)
CHARS_PER_TOKEN = 4
FILLER = (
    "The following padding text is used only to increase input context length "
    "for baseline benchmarking and should be ignored. "
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
        cfg[key.strip()] = val
        os.environ.setdefault(key.strip(), val)
    return cfg


def pad_prompt(prompt: str, target_input_tokens: int) -> str:
    """Pad prompt upward toward target_input_tokens using filler text."""
    current_est = max(1, len(prompt) // CHARS_PER_TOKEN)
    if current_est >= target_input_tokens:
        return prompt
    need_tokens = target_input_tokens - current_est
    need_chars = need_tokens * CHARS_PER_TOKEN
    repeats = (need_chars // len(FILLER)) + 1
    padding = (FILLER * repeats)[:need_chars]
    return f"{padding}\n\n{prompt}"


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
    model: str
    api_base: str
    config: dict[str, Any]
    context_lengths: list[int]
    results: list[RequestResult] = field(default_factory=list)


def stream_chat(
    api_base: str,
    model: str,
    content: str,
    max_tokens: int,
) -> tuple[float | None, float, dict[str, int | None], str | None]:
    """
    Returns (ttft_ms, total_elapsed_ms, usage_dict, error).
    ttft_ms is None if no token received.
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
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
        with urllib.request.urlopen(req, timeout=600) as resp:
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

                if "usage" in event and event["usage"]:
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
    except Exception as e:  # noqa: BLE001
        return None, (time.perf_counter() - t0) * 1000, usage, str(e)

    total_ms = (time.perf_counter() - t0) * 1000
    return ttft, total_ms, usage, None


def main() -> int:
    load_config_env()

    model = os.environ.get("SERVED_MODEL_NAME", "qwen3-32b-awq")
    api_base = os.environ.get("BENCH_API_BASE", "http://127.0.0.1:8000/v1")
    max_tokens = int(os.environ.get("BENCH_MAX_TOKENS", "256"))
    context_lengths = [
        int(x.strip())
        for x in os.environ.get("BENCH_CONTEXT_LENGTHS", "256,2048,8000").split(",")
        if x.strip()
    ]

    if not PROMPTS_FILE.exists():
        print(f"ERROR: {PROMPTS_FILE} not found", file=sys.stderr)
        return 1

    prompts_data = json.loads(PROMPTS_FILE.read_text())
    prompts = prompts_data["prompts"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"bench_{ts}.json"

    run = BenchRun(
        timestamp=ts,
        model=model,
        api_base=api_base,
        config={
            "TP": os.environ.get("TP"),
            "MAXLEN": os.environ.get("MAXLEN"),
            "GPU_MEM_UTIL": os.environ.get("GPU_MEM_UTIL"),
            "BENCH_MAX_TOKENS": max_tokens,
            "BENCH_CONTEXT_LENGTHS": context_lengths,
        },
        context_lengths=context_lengths,
    )

    print(f"Benchmark: model={model} contexts={context_lengths} prompts={len(prompts)}")
    print(f"API: {api_base}")
    print("Concurrency: 1 (sequential)\n")

    for ctx in context_lengths:
        print(f"--- context target ~{ctx} tokens ---")
        for p in prompts:
            pid = p["id"]
            padded = pad_prompt(p["text"], ctx)
            print(f"  {pid} @ ctx={ctx} ... ", end="", flush=True)

            sampler = VramSampler()
            sampler.start()
            ttft_ms, total_ms, usage, err = stream_chat(
                api_base, model, padded, max_tokens
            )
            peak_mib = sampler.stop()

            decode_tok_s: float | None = None
            comp = usage.get("completion_tokens")
            if ttft_ms is not None and comp and comp > 0:
                decode_sec = max(1e-9, (total_ms - ttft_ms) / 1000)
                decode_tok_s = comp / decode_sec

            rr = RequestResult(
                prompt_id=pid,
                category=p.get("category", ""),
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
            run.results.append(rr)

            if err:
                print(f"ERROR: {err}")
            elif ttft_ms is not None:
                dts = f"{decode_tok_s:.1f}" if decode_tok_s else "n/a"
                print(f"TTFT={ttft_ms:.0f}ms decode={dts} tok/s VRAM={peak_mib}MiB")
            else:
                print("no tokens received")

            time.sleep(0.5)

    out_path.write_text(json.dumps({"run": asdict(run)}, indent=2))
    print(f"\nWrote {out_path}")
    print("Next: python summarize.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
