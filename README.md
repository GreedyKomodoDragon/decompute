# Decompute

Small localhost prototype for decentralized model inference:

```text
client -> coordinator -> worker -> local Qwen model -> worker -> coordinator -> client
```

The coordinator only understands HTTP and the shared protocol. Candle, tokenization, model files, and device selection are isolated to the inference/worker side.

## Workspace crates

| Crate | Responsibility |
| --- | --- |
| `protocol` | Shared, serializable network types: model capabilities, worker registration and heartbeats, generation requests/responses, API errors, hardware data, and manifests. It has no HTTP or inference dependency. |
| `inference` | Local model runtime. It loads the Qwen config, tokenizer, and safetensors weights; selects a compatible execution dtype for the device; generates tokens; calculates model manifests; and reports hardware basics. This is the only library crate that knows Candle. |
| `inference-example` | Small executable for proving local inference before networking. It loads `./models/tiny-model` and prints generated text. |
| `worker` | Process that owns a complete local model. Its Axum server exposes health, capabilities, generation, SSE streaming, and draining endpoints. A dedicated OS thread owns the model and receives jobs over a channel, so blocking inference never occupies Tokio worker threads. It registers with the coordinator and heartbeats every five seconds. |
| `coordinator` | Inference-library-free Axum service. It stores worker records, expires stale heartbeats, selects the least-busy eligible worker with an exact model match, forwards requests with Reqwest, and proxies SSE streams. |
| `client` | Small CLI client for the coordinator's non-streaming public generation endpoint. `curl` remains the simplest way to exercise the HTTP API. |

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
curl http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-model","prompt":"Why is the sky blue?","max_tokens":100}'
```

The public endpoint creates a UUID if `request_id` is omitted and responds with the selected worker, text, and nested token usage. Inspect the registry with `curl http://127.0.0.1:8000/workers`.

For streaming, use `curl -N` against `/v1/generate/stream` with the same JSON body. It proxies worker SSE frames and ends with `data: [DONE]`.

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
