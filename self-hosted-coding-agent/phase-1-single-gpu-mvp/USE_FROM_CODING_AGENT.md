# Use the self-hosted model from a coding agent (Cursor / aider / opencode)

Instead of only running SWE-bench, you can point an interactive coding agent at the same vLLM
endpoint and use the model for real coding tasks. The server is already OpenAI-compatible, so any
tool that speaks the OpenAI `/v1` API can talk to it.

This guide assumes the model is served by [`02_serve_vllm.sh`](02_serve_vllm.sh) and the box was
launched by [`01_infra_setup.sh`](01_infra_setup.sh).

---

## 0. The three facts every client needs

| What | Value (from the serve script) |
|------|-------------------------------|
| Base URL (on the box) | `http://127.0.0.1:8000/v1` |
| Served model name | `qwen3.6-35b-a3b-fp8` |
| API key | any non-empty string (vLLM ignores it, e.g. `dummy`) |

The model name must match **exactly** what the server reports. Confirm it any time with:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Reasoning is ON (`--reasoning-parser qwen3`) and tool-calling is ON (`--tool-call-parser qwen3_xml`),
so agent/tool-use flows work.

---

## 1. Make the endpoint reachable from your Mac

Port 8000 is **not** open publicly (and shouldn't be). There are two ways to reach it, and which one
you use depends on the client:

### Option A — SSH tunnel (plain HTTP on localhost) — for aider & opencode

Load the instance details written by the infra script and open a tunnel:

```bash
cd self-hosted-coding-agent/phase-1-single-gpu-mvp
source instance.env   # exports INSTANCE_PUBLIC_IP, SSH_KEY_PATH, SSH_USER, ...

ssh -i "$SSH_KEY_PATH" -L 8000:localhost:8000 "$SSH_USER@$INSTANCE_PUBLIC_IP"
```

Leave that terminal open. Now `http://127.0.0.1:8000/v1` on your Mac forwards to the box. This is all
aider and opencode need.

### Option B — Public HTTPS tunnel — required for Cursor

Cursor **cannot** use a localhost/HTTP endpoint. All requests are proxied through Cursor's servers,
so it needs a **publicly reachable HTTPS URL**. Expose port 8000 with a tunnel that gives you HTTPS.

Run one of these **on the box** (SSH in first), pointing at the local vLLM port:

```bash
# Cloudflare (no signup for a quick tunnel):
cloudflared tunnel --url http://localhost:8000
# -> prints a https://<random>.trycloudflare.com URL

# or ngrok (needs a free account + authtoken):
ngrok http 8000
# -> prints a https://<random>.ngrok-free.app URL
```

Copy the printed `https://...` URL; you'll append `/v1` to it in Cursor.

> Security note: vLLM has no auth by default, so a public tunnel exposes the model to anyone with the
> URL. For anything beyond a quick test, add `--api-key <secret>` to the `vllm serve` command in
> `02_serve_vllm.sh` and use that secret as the API key in the client. Tear the tunnel down when done.

---

## 2. Cursor

1. Start a **public HTTPS tunnel** (Option B) and copy the URL.
2. Cursor → **Settings → Models** (Cursor Settings, not VS Code settings).
3. Scroll to the **OpenAI API Key** section:
   - Enter any non-empty string as the API key (e.g. `dummy`, or your `--api-key` secret if you set one).
   - Toggle **Override OpenAI Base URL** on and paste `https://<your-tunnel>/v1` (note the `/v1`).
4. Click **Add Model** and enter the exact served name: `qwen3.6-35b-a3b-fp8`.
5. Deselect the other (cloud) models so Cursor routes to yours, then **Verify**.

Notes / limitations:
- Works for **Chat, Cmd+K, and Agent**; **Tab autocomplete stays cloud-only** regardless of config.
- If Verify fails with a connection error, set **Settings → Network → HTTP Compatibility Mode** to
  **HTTP/1.1**.
- Base URL must be **HTTPS** — a plain `http://` tunnel or `localhost` will not work.

---

## 3. aider

aider calls the endpoint directly from your machine, so the **SSH tunnel (Option A)** is enough — no
public HTTPS needed.

Quick one-off:

```bash
pip install -U aider-install && aider-install   # or: pip install aider-chat

aider \
  --model openai/qwen3.6-35b-a3b-fp8 \
  --openai-api-base http://127.0.0.1:8000/v1 \
  --openai-api-key dummy
```

Persistent config — create `~/.aider.conf.yml` (or one at your repo root):

```yaml
model: openai/qwen3.6-35b-a3b-fp8
openai-api-base: http://127.0.0.1:8000/v1
openai-api-key: dummy
edit-format: diff
auto-commits: false
```

To silence the "unknown context window / cost" warning and cap output, add a
`~/.aider.model.settings.yml`:

```yaml
- name: openai/qwen3.6-35b-a3b-fp8
  edit_format: diff
  use_repo_map: true
  extra_params:
    max_tokens: 32000
```

> The `openai/` prefix tells aider (via litellm) to treat this as a generic OpenAI-compatible
> endpoint. Keep prompts/context under the server's `--max-model-len` (currently `32768`); if you hit
> context overflows, raise `--max-model-len` in `02_serve_vllm.sh` (KV budget permitting) or lower
> `max_tokens` here.

---

## 4. opencode

opencode also calls the endpoint directly, so the **SSH tunnel (Option A)** is enough.

Edit `~/.config/opencode/opencode.json` (create it if missing):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Self-hosted Qwen3.6 (vLLM)",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dummy"
      },
      "models": {
        "qwen3.6-35b-a3b-fp8": {
          "name": "Qwen3.6 35B A3B FP8"
        }
      }
    }
  },
  "model": "vllm/qwen3.6-35b-a3b-fp8",
  "small_model": "vllm/qwen3.6-35b-a3b-fp8"
}
```

Then launch `opencode`, run `/models`, and pick **Self-hosted Qwen3.6 (vLLM)**. The provider id
(`vllm`) plus the model key (`qwen3.6-35b-a3b-fp8`) form the `vllm/qwen3.6-35b-a3b-fp8` selector.

> If opencode complains about missing auth, either keep `options.apiKey` as above or add a
> `{ "vllm": { "type": "api", "key": "dummy" } }` entry to `~/.config/opencode/auth.json`.

---

## 5. Context limits and compaction (important)

The server is served at `--max-model-len 32768`. **vLLM never compacts** — if a request's
tokens exceed that, it returns HTTP 400 (`The prompt is too long: N, model maximum context
length: 32768`), which clients surface as a `ContextWindowExceededError`. Whether that error is
*handled* (by summarizing/compacting history) or is *fatal* is entirely a **client-side** decision:

| Client | Behavior on overflow |
|--------|----------------------|
| `mini-swe-agent` (the SWE-bench harness) | **No compaction — the task just fails.** It is a minimal agent with no context management. This is exactly why `astropy-14182` failed in `RESULTS.md`: it hit 32768, the server 400'd, and the harness has no fallback. |
| **aider** | Recursive summarization of chat history in a background thread, via a separate "weak model", at a soft `max_chat_history_tokens` limit. |
| **opencode** | Auto-compaction **on by default**: summarizes to a checkpoint + keeps the most recent ~8k tokens, triggered before overflow (configurable via the `compaction` block). |
| **Cursor** | Manages/summarizes context on its own servers in Agent mode. |

Two gotchas specific to this self-hosted endpoint:

1. **aider's summarizer must also point here.** Its background summarization uses a *separate*
   "weak model"; if unset it falls back to a default (OpenAI) model that isn't configured. Point it
   at the same endpoint:

   ```bash
   aider --model openai/qwen3.6-35b-a3b-fp8 \
         --weak-model openai/qwen3.6-35b-a3b-fp8 \
         --openai-api-base http://127.0.0.1:8000/v1 --openai-api-key dummy
   ```

2. **Compaction can't save you if the *fixed* part overflows.** System prompt + tool schemas + a
   single huge file read + the latest turn can exceed 32768 before there's any history to summarize;
   then even aider/opencode surface a provider overflow. The durable fix is to **raise
   `--max-model-len`** in `02_serve_vllm.sh` (KV budget permitting on the 48 GB L40S) or feed less
   per turn.

opencode compaction stays on by default; to tune it, add to `opencode.json`:

```json
{
  "compaction": { "auto": true, "reserved": 20000 }
}
```

---

## 6. Sanity check before wiring up a client

With the tunnel open, confirm the endpoint answers a normal chat request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "qwen3.6-35b-a3b-fp8",
    "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}]
  }' | python3 -m json.tool
```

If that returns a completion, every client above will work once configured.

---

## Troubleshooting

- **`connection refused` / empty `/v1/models`** — the server isn't up yet. On the box, wait for
  `Application startup complete` from `02_serve_vllm.sh`, and confirm the tunnel/SSH session is alive.
- **Model not found** — the name in the client must equal the server's id exactly
  (`qwen3.6-35b-a3b-fp8`); re-check with `curl .../v1/models`.
- **Context window exceeded** on long agent sessions — the server is at `--max-model-len 32768` and
  does not compact; see §5 for how each client handles this. Raise `--max-model-len` (KV budget
  permitting on 48 GB), lower the client's `max_tokens`, or rely on client-side compaction.
- **Cursor "Verify" fails** — base URL must be HTTPS (use Option B), append `/v1`, and try HTTP/1.1
  compatibility mode.
- **Slow first request** — cold start includes `torch.compile`; single-stream decode is ~75 tok/s on
  the L40S at FP8, so expect interactive-but-not-instant latency.
