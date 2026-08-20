# Decompute

Small localhost prototype for decentralized model inference:

```text
client -> coordinator -> worker -> local Qwen model -> worker -> coordinator -> client
```

The coordinator only understands HTTP and the shared protocol. GGUF model loading, embedded chat templates, tokenization, and device selection are isolated to the worker/SDK side.

## Project status

Decompute is an experimental local-inference prototype, not production infrastructure. Its supported development platform is Apple Silicon macOS with Metal, and its public API and on-disk formats may change before a stable release. It intentionally does not implement authentication, authorization, TLS, public-network hardening, payments, P2P discovery, or multi-tenant isolation.

Use it on localhost or a trusted private network only. See [SECURITY.md](SECURITY.md) before binding a coordinator or worker to a non-loopback address.

## Workspace crates

| Crate | Responsibility |
| --- | --- |
| `protocol` | Shared, serializable network types: model capabilities, worker registration and heartbeats, chat/generation requests/responses, API errors, hardware data, and manifests. It has no HTTP or inference dependency. |
| `inference-example` | Small executable for proving local GGUF inference before networking. It intentionally loads a local GGUF file directly, without provisioning or networking. |
| `worker` | Process that owns a complete local model. Its Axum server exposes health, capabilities, generation, SSE streaming, and draining endpoints. The SDK owns blocking llama.cpp inference on a dedicated OS thread, so Tokio remains free for HTTP and heartbeats. |
| `coordinator` | Inference-library-free Axum service. It exposes an OpenAI Chat Completions-compatible API, stores worker records, expires stale heartbeats, selects the least-busy eligible worker with an exact model match, and proxies private inference requests. |
| `client` | Small CLI client for the coordinator's OpenAI-compatible endpoint. `curl` or OpenCode are the preferred API clients. |
| `harness` | Native macOS Apple-Silicon egui chat client. It uses only OpenAI-compatible models/chat-completions/SSE endpoints and has no inference or protocol dependency. |
| `decompute-models` | Curated model catalog, Hugging Face cache/download resolution, checksum verification, and model provenance. It has no inference, coordinator, or UI dependency. |

## In-process SDK foundations

The workspace now separates reusable local-inference concerns from network transport:

| Crate | Responsibility |
| --- | --- |
| `decompute-core` | Transport-neutral chat, tool-call, model-manifest, and hardware types. |
| `decompute-sdk` | Public async-facing GGUF model handle backed by a dedicated model thread and progressive generation events. |
| `decompute-llama` | Opt-in GGUF/llama.cpp runtime: metadata inspection, model loading, embedded-template rendering, token generation, and optional Metal compilation. |
| `decompute-models` | Reusable model-provisioning layer. It resolves a curated catalog entry to a verified local GGUF file before the SDK loads it. |

`protocol` re-exports the domain types from `decompute-core` for source compatibility, but continues to own HTTP worker/coordinator payloads and lifecycle state. Neither `protocol` nor `coordinator` depends on a model runtime.

The llama.cpp binding compiles native C++ code. On this Mac, build it with Homebrew LLVM rather than the incomplete Command Line Tools C++ driver:

```bash
CC="$(brew --prefix llvm)/bin/clang" CXX="$(brew --prefix llvm)/bin/clang++" \
  cargo check -p decompute-sdk --features llama
```

Use `llama-metal` instead of `llama` to compile llama.cpp with Metal support. The worker selects it with `--device metal` or probes it with `--device auto`. Qwen GGUF tool calls are supported; embeddings, reranking, vision, audio transcription, and GBNF generation remain future runtime additions.

The process boundary is deliberate: moving a worker to another machine only changes its bind/advertise address; neither the coordinator nor protocol needs to know how the model is executed.

## Prerequisites

- Apple Silicon macOS with Metal. This prototype's supported local platform is macOS/Metal only.
- Rust 1.94.0, selected automatically by `rust-toolchain.toml`
- [just](https://github.com/casey/just), installed with `brew install just`
- Internet access for the first worker start. The curated Qwen2.5 Q4_K_M downloads are about 469 MB (0.5B) and 1.12 GB (1.5B).

Workers automatically resolve their selected curated model from Hugging Face before loading it. The catalog pins the repository revision and SHA-256, and the worker verifies the complete file before it can register. No Hugging Face CLI is required.

The Hugging Face cache defaults to the standard `~/.cache/huggingface/hub` location. Set `HF_HOME` or `HF_HUB_CACHE` to relocate it. For gated/private catalog entries, Hugging Face's standard credential resolution is used, including `HF_TOKEN` and its local token store; Decompute never persists, logs, forwards, or exposes credentials.

## Run locally

First check standalone inference:

```bash
CC="$(brew --prefix llvm)/bin/clang" CXX="$(brew --prefix llvm)/bin/clang++" cargo run -p inference-example
```

Then use four terminals:

```bash
just coordinator
```

```bash
just worker-a
```

```bash
just worker-b
```

Both default workers load Qwen2.5 0.5B, so repeated requests for that model exercise exact-match scheduling, load balancing, and worker redundancy. After both workers are ready, `GET /v1/models` and **Connect / refresh models** in the harness show the loaded model ID.

In a fourth terminal, start the native macOS chat harness:

```bash
just harness
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b-instruct-q4-k-m","messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":100}'
```

The public API is OpenAI Chat Completions-compatible: `POST /v1/chat/completions` and `GET /v1/models`. The coordinator selects a worker and proxies a private request; clients never receive worker addresses. Inspect the internal registry with `curl http://127.0.0.1:8000/workers`.

The optional `X-Decompute-Session-Id` request header accepts a client-generated
UUID. Repeated requests with the same model and UUID prefer the same eligible
worker for up to 15 minutes. This is best-effort routing metadata, never a
correctness requirement. Workers retain local llama.cpp KV state with
`--session-cache-capacity` (default `1`; `0` disables reuse); entries are
process-local, ephemeral, LRU-evicted, and idle-expire after 15 minutes. The
worker also supports `--session-cache-min-tokens` (default `100`),
`--session-cache-max-bytes` (default `0`, unlimited; non-zero values reserve a
conservative full-context KV estimate per cached entry), and
`--session-cache-slot-wait-secs` (default `30`) for same-session serialization.
Cache checkpoints are taken at the rendered prompt boundary;
generated output is never added to the reusable prompt prefix. Cancellation,
errors, template/tool changes, prefix mismatches, and context overflow discard
the affected entry.
Affinity and caching are not encryption, authentication, or confidentiality:
the coordinator still sees plaintext requests, and privileged local processes
may inspect worker memory.

Workers can push privacy-safe logs and session-cache counters to an
OpenTelemetry Collector by setting `OTEL_EXPORTER_OTLP_ENDPOINT` (and
optionally `OTEL_SERVICE_NAME`). The standard OTLP/HTTP environment variables
control signal-specific endpoints, headers, and timeout behavior. Cache logs
contain only event categories and aggregate token counts; they never contain
session UUIDs, prompt text, or token values. Without an OTLP endpoint, logs
remain local as usual.

### Native macOS harness

`just harness` starts a small egui desktop client for Apple Silicon macOS. It connects to an external OpenAI-compatible endpoint—by default `http://127.0.0.1:8000`—so its wire protocol is not coupled to Decompute. Use **Connect / refresh models** to discover the model advertised by the coordinator, select it, then chat with progressive SSE output.

The harness sends no hidden system message. Its **System harness** editor is disabled and empty by default; enabling it adds exactly the visible text as the leading `system` message. It never executes tools.

Chats and settings persist locally. Delete a chat from its sidebar row; Decompute asks for confirmation, and deleting an active chat stops its generation before removing the local history. The context indicator is intentionally only an estimate (characters divided by four), because a generic OpenAI-compatible model listing does not expose tokenizer or context metadata. The harness never silently drops or summarizes history: remove messages or clear a chat yourself when the estimate exceeds the editable context budget. Press **Stop** to abandon a stream; the coordinator then cancels the corresponding worker generation.

The Harness has interaction tests for its accessible controls. Run them with `cargo test -p harness`.

Each worker checks its Hugging Face cache, downloads the pinned file if necessary, verifies its SHA-256, then starts inference. Workers load exactly one selected model, and share the cache safely. To test multi-model discovery instead of same-model scheduling, run `just worker-b-1-5b` in place of `just worker-b`.

Override either recipe without changing the harness or coordinator:

```bash
WORKER_A_MODEL_ID=qwen2.5-1.5b-instruct-q4-k-m just worker-a
WORKER_B_MODEL_ID=qwen2.5-1.5b-instruct-q4-k-m just worker-b
```

To run offline with an already-downloaded copy of that exact catalog model, use `--model-path`; it is still GGUF and SHA-256 verified:

```bash
CC="$(brew --prefix llvm)/bin/clang" CXX="$(brew --prefix llvm)/bin/clang++" cargo run -p worker --features metal -- \
  --device metal --port 9001 --node-id worker-a \
  --coordinator http://127.0.0.1:8000 \
  --model qwen2.5-0.5b-instruct-q4-k-m \
  --model-path ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Workers default to `--context-tokens auto`, which reads the GGUF model's trained context length from its metadata before loading inference tensors. Pass `--context-tokens <non-zero-value>` to explicitly override that capacity. Larger contexts consume more unified memory for the KV cache, so configure every OpenAI-compatible client with a context limit no greater than the capacity reported in the worker startup log. The included OpenCode and Pi examples advertise the 32,768-token context stored by the curated Qwen models.

The catalog is embedded from [`crates/decompute-models/catalog.toml`](crates/decompute-models/catalog.toml). Add a model by creating a pinned `gguf` entry with its repository, full revision, filename, SHA-256, architecture, and quantization. The harness does not need changing: it discovers only the model IDs loaded by workers through `GET /v1/models`.

### Chat messages and model templates

Requests use standard OpenAI `messages`:

```json
{"model":"qwen2.5-0.5b-instruct-q4-k-m","messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":100}
```

For system instructions and chat history, pass structured messages:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b-instruct-q4-k-m",
    "messages": [
      {"role": "system", "content": "Answer in one sentence."},
      {"role": "user", "content": "Why is the sky blue?"}
    ],
    "max_tokens": 100
  }'
```

At model load, llama.cpp reads the chat template embedded in the GGUF and applies it to normalized messages with a generation prompt. This avoids a hard-coded Qwen prompt wrapper and makes the worker portable across GGUF models that package their own template.

For safety, templates are loaded only once from the model-local override directory; they have no arbitrary filesystem loader or host callbacks. Multimodal message content is intentionally unsupported.

### Tool-call proposals

The HTTP API accepts OpenAI-style tool definitions and tool-history messages. For Qwen GGUF models, the worker selects its built-in Qwen tool template, renders the supplied schemas, parses `<tool_call>` JSON blocks, and returns standard OpenAI tool calls. Neither component executes a tool.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b-instruct-q4-k-m",
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

The client owns the execution loop: validate and execute each proposed call in its own trusted environment, then submit an assistant tool-call message and one `tool` message per result. The worker renders the tool history back into Qwen's expected format. For another model family, add a model-aware template and parser without changing the coordinator protocol.

### OpenCode

Copy [`examples/opencode.json`](examples/opencode.json) into your OpenCode project configuration (or merge its `provider.decompute` entry with your existing configuration). With the coordinator and at least one worker running, use:

```bash
opencode run --model decompute/qwen2.5-0.5b-instruct-q4-k-m "Explain this repository and suggest one small improvement."
```

Use the model ID returned by `GET /v1/models` if your workers advertise a different model. OpenCode communicates only with the coordinator. It executes file, shell, and other coding tools on the OpenCode machine; Decompute workers only receive private model-inference requests.

### Pi

Pi can use the same OpenAI-compatible coordinator. Merge the `decompute` provider from [`examples/pi-models.json`](examples/pi-models.json) into `~/.pi/agent/models.json`, then make a deliberately small no-tools request while testing the initial model setup:

```bash
pi --provider decompute --model qwen2.5-0.5b-instruct-q4-k-m \
  --no-session --no-tools \
  --system-prompt 'Reply concisely.' \
  --print 'Reply with exactly: hello'
```

The sample declares a 2,048-token context window and 96-token output limit. Those limits keep Pi's initial tests within the practical range of CPU inference; increase them only after an accelerator path is available.

### Models and templates

GGUF metadata selects both the model architecture and its embedded default chat template. Adding a GGUF-compatible model normally only requires pointing a worker at another `.gguf` file; no coordinator or protocol change is needed.

For model-specific experimentation, add named MiniJinja overrides beside the model:

```text
models/
└── qwen2.5-0.5b-instruct-q4_k_m.gguf.templates/
    ├── rag.jinja
    └── partials/
        └── context.jinja
```

Top-level `*.jinja` files are selectable by filename without the extension. They may `include` or `import` files under subdirectories. Select one through Decompute's optional `template` request field:

```json
{
  "model": "qwen2.5-0.5b-instruct-q4-k-m",
  "template": "rag",
  "messages": [{"role": "user", "content": "Summarize this context."}]
}
```

Templates are loaded once at worker startup from this model-local directory; they cannot load arbitrary host files. With no explicit override, the GGUF embedded template remains the default. Qwen tool requests automatically select the built-in `qwen-tools` template.

For streaming, add `"stream": true` to `POST /v1/chat/completions`. Visible text is forwarded from the worker through the coordinator as it is generated, using OpenAI-style Server-Sent Events and a final `data: [DONE]`. Qwen tool-call markup is withheld until generation completes, then emitted as structured OpenAI tool-call chunks.

For a model loaded by both accelerated and CPU workers, the coordinator prefers Metal (and future CUDA) workers, then uses active request count and worker ID as deterministic tie-breakers. If the client disconnects, the coordinator sends a private cancellation request to the selected worker; the worker stops at the next safe llama.cpp execution boundary and only then releases its capacity.

Drain a worker without killing an in-flight request:

```bash
curl -X POST http://127.0.0.1:9001/drain
```

## Trusted private-network worker

> **Security warning:** this mode is for a trusted private network only. The coordinator and worker APIs currently have no authentication, authorization, TLS, rate limiting, or tenant isolation. Do not expose either process to the internet or an untrusted LAN.

To experiment with a worker on another trusted machine, run the coordinator on a reachable interface and give the worker a reachable advertised URL:

Run the coordinator on a reachable interface, then run a worker with a reachable advertised URL:

```bash
cargo run -p coordinator -- --bind 0.0.0.0
CC="$(brew --prefix llvm)/bin/clang" CXX="$(brew --prefix llvm)/bin/clang++" cargo run -p worker -- --bind 0.0.0.0 --port 9001 --advertise-address http://worker-host:9001 --node-id worker-b --coordinator http://coordinator-host:8000 --model qwen2.5-0.5b-instruct-q4-k-m
```

No transport or protocol changes are needed. Before any public-network deployment, add authentication and TLS at a minimum; those controls are explicitly outside this prototype’s current scope.

## Devices and identity

The worker reads GGUF metadata to identify the model architecture and computes a SHA-256 manifest from the model file. Two workers using the same file therefore advertise the same manifest ID. It runs a one-token smoke test before registering, so a native-backend failure never creates an apparently healthy worker.

Apple Metal is the supported local acceleration target and does not alter the network protocol. Build with the feature and explicitly select Metal:

```bash
CC="$(brew --prefix llvm)/bin/clang" CXX="$(brew --prefix llvm)/bin/clang++" cargo run -p worker --features metal -- \
  --device metal \
  --port 9001 \
  --node-id worker-a \
  --coordinator http://127.0.0.1:8000 \
  --model qwen2.5-0.5b-instruct-q4-k-m
```

`--device auto` probes Metal when compiled, verifies it with the smoke test, then falls back to CPU if the probe fails. An explicit `--device metal` never falls back silently: it fails startup with the full compatibility error. CUDA is not implemented yet; its protocol enum is reserved for later worker support.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull-request expectations. Run the full local validation before opening a pull request:

```bash
just test
cargo fmt --check
CC="$(brew --prefix llvm)/bin/clang" \
CXX="$(brew --prefix llvm)/bin/clang++" \
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Install the shared Git checks after cloning:

```bash
brew install lefthook
lefthook install
```

Lefthook runs formatting, strict Clippy, and an all-feature workspace check before commits; it runs the full test suite before pushes.

Please report security-sensitive issues privately as described in [SECURITY.md](SECURITY.md), rather than through a public issue.
