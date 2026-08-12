# Decompute

Small localhost prototype for decentralized model inference:

```text
client -> coordinator -> worker -> local Qwen model -> worker -> coordinator -> client
```

The coordinator only understands HTTP and the shared protocol. Candle, tokenization, model files, and device selection are isolated to the inference/worker side.

## Workspace crates

| Crate | Responsibility |
| --- | --- |
| `protocol` | Shared, serializable network types: model capabilities, worker registration and heartbeats, chat/generation requests/responses, API errors, hardware data, and manifests. It has no HTTP or inference dependency. |
| `inference` | Local model runtime. It loads the Qwen config, tokenizer, chat template, and safetensors weights; selects a compatible execution dtype for the device; generates tokens; calculates model manifests; and reports hardware basics. This is the only library crate that knows Candle. |
| `inference-example` | Small executable for proving local inference before networking. It loads `./models/tiny-model` and prints generated text. |
| `worker` | Process that owns a complete local model. Its Axum server exposes health, capabilities, generation, SSE streaming, and draining endpoints. A dedicated OS thread owns the model and receives jobs over a channel, so blocking inference never occupies Tokio worker threads. It registers with the coordinator and heartbeats every five seconds. |
| `coordinator` | Inference-library-free Axum service. It exposes an OpenAI Chat Completions-compatible API, stores worker records, expires stale heartbeats, selects the least-busy eligible worker with an exact model match, and proxies private inference requests. |
| `client` | Small CLI client for the coordinator's OpenAI-compatible endpoint. `curl` or OpenCode are the preferred API clients. |

The process boundary is deliberate: moving a worker to another machine only changes its bind/advertise address; neither the coordinator nor protocol needs to know how the model is executed.

## Prerequisites

- Rust stable (this workspace was built with Rust 1.94)
- A local copy of `Qwen/Qwen2.5-0.5B-Instruct` in safetensors format (about 1 GB)
- The Hugging Face CLI, for example: `pipx install huggingface_hub`

Download the model once:

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/tiny-model
```

The directory must contain `config.json`, `tokenizer.json`, and `model.safetensors`.

## Run locally

First check standalone inference:

```bash
cargo run -p inference-example
```

Then use four terminals:

```bash
cargo run -p coordinator
```

```bash
cargo run -p worker -- --port 9001 --node-id worker-a --coordinator http://127.0.0.1:8000 --model ./models/tiny-model
```

```bash
cargo run -p worker -- --port 9002 --node-id worker-b --coordinator http://127.0.0.1:8000 --model ./models/tiny-model
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-model","messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":100}'
```

The public API is OpenAI Chat Completions-compatible: `POST /v1/chat/completions` and `GET /v1/models`. The coordinator selects a worker and proxies a private request; clients never receive worker addresses. Inspect the internal registry with `curl http://127.0.0.1:8000/workers`.

### Chat messages and model templates

Requests use standard OpenAI `messages`:

```json
{"model":"tiny-model","messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":100}
```

For system instructions and chat history, pass structured messages:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tiny-model",
    "messages": [
      {"role": "system", "content": "Answer in one sentence."},
      {"role": "user", "content": "Why is the sky blue?"}
    ],
    "max_tokens": 100
  }'
```

At model load, the worker reads the Hugging Face `chat_template` from `tokenizer_config.json` and compiles it with MiniJinja. It renders the normalized messages with `add_generation_prompt: true`, then passes the rendered text to that model's `tokenizer.json`. This replaces the prior hard-coded Qwen prompt wrapper. The template renderer is independent of Candle model execution, so a later architecture adapter can reuse it for another supported model family.

For safety, templates have no filesystem loader or host callbacks. Multimodal message content is intentionally unsupported.

### Tool-call proposals (Qwen)

The Qwen2.5 template bundled with the selected model supports OpenAI-style function definitions. Send a `tools` array with structured messages; Qwen can return one or more proposed calls. The worker parses Qwen's `<tool_call>…</tool_call>` output and the coordinator returns it unchanged. Neither component executes a tool.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tiny-model",
    "messages": [{"role": "user", "content": "What time is it in London?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_time",
        "description": "Get the current time in an IANA timezone.",
        "parameters": {
          "type": "object",
          "properties": {"timezone": {"type": "string"}},
          "required": ["timezone"]
        }
      }
    }],
    "max_tokens": 100
  }'
```

A successful proposal has `choices[0].finish_reason: "tool_calls"` and `choices[0].message.tool_calls`. Each call has a deterministic request-scoped ID, a function name, and JSON-encoded arguments:

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {"tool_calls": [{
    "id": "<request-id>-0",
    "type": "function",
    "function": {"name": "get_time", "arguments": "{\"timezone\":\"Europe/London\"}"}
  }]}}
  ]
}
```

The client owns the execution loop: validate and execute each proposed call in its own trusted environment, then submit a follow-up OpenAI `messages` request containing the assistant `tool_calls` and one `tool` message per result. The Qwen template already renders that history. Tool names must be unique ASCII letters, digits, `_`, or `-`; each `parameters` value must be a JSON object.

### OpenCode

Copy [`examples/opencode.json`](examples/opencode.json) into your OpenCode project configuration (or merge its `providers.decompute` entry with your existing configuration). With the coordinator and at least one worker running, use:

```bash
opencode run --model decompute/tiny-model "Explain this repository and suggest one small improvement."
```

Use the model ID returned by `GET /v1/models` if your workers advertise a different model. OpenCode communicates only with the coordinator. It executes file, shell, and other coding tools on the OpenCode machine; Decompute workers only receive private model-inference requests.

### Providers and named templates

The worker detects the model family from `config.json`'s `model_type` and uses a matching inference provider. Qwen2 is the initial registered provider. Adding a family such as Llama or Mistral means implementing one provider module that loads its Candle model and returns rank-one next-token logits; tokenization, sampling, HTTP, scheduling, and SSE remain unchanged.

Templates are independent from providers. A model directory can supply them in either Hugging Face format:

- `chat_template.jinja` takes precedence as the `default` template.
- Otherwise, `tokenizer_config.json`'s `chat_template` may be one string (`default`) or an object of named templates. Named templates can use MiniJinja `include` and `import` to share subtemplates.

Choose a named template explicitly when the model provides one:

```json
{
  "model": "tiny-model",
  "template": "rag",
  "messages": [{"role": "user", "content": "Summarize this context."}]
}
```

If no `template` is supplied, the worker uses `default`. Unknown names return a clear error listing the templates packaged by that model.

For streaming, add `"stream": true` to `POST /v1/chat/completions`. The coordinator returns OpenAI-style Server-Sent Events and ends with `data: [DONE]`.

Drain a worker without killing an in-flight request:

```bash
curl -X POST http://127.0.0.1:9001/drain
```

## Another physical machine

Run the coordinator on a reachable interface, then run a worker with a reachable advertised URL:

```bash
cargo run -p coordinator -- --bind 0.0.0.0
cargo run -p worker -- --bind 0.0.0.0 --port 9001 --advertise-address http://worker-host:9001 --node-id worker-b --coordinator http://coordinator-host:8000 --model ./models/tiny-model
```

No transport or protocol changes are needed; this prototype intentionally does not add authentication, NAT traversal, retries, payments, or P2P discovery.

## Devices and identity

CPU is the default. The worker inspects the safetensors header to determine its stored weight dtype. F32 weights remain F32; BF16/F16 weights are promoted to F32 on CPU because Candle CPU matmul cannot execute BF16. The worker reports basic architecture/RAM information and hashes `config.json`, `tokenizer.json`, and `model.safetensors` into a deterministic manifest ID. It currently routes the user-friendly `tiny-model` alias.

Apple Metal is optional and does not alter the network protocol. Build with the feature and explicitly select Metal:

```bash
cargo run -p worker --features metal -- \
  --device metal \
  --port 9001 \
  --node-id worker-a \
  --coordinator http://127.0.0.1:8000 \
  --model ./models/tiny-model
```

`--device auto` uses Metal when the feature is compiled and initialization succeeds, otherwise it falls back to CPU. CUDA is not implemented yet; its protocol enum is reserved for later worker support.
