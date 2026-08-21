# decompute-core

Transport-neutral types shared by the Decompute SDK and runtimes.

This crate contains chat messages, tool definitions and calls, generation
results, model manifests, hardware information, and session-cache types. It
does not load models, execute inference, or perform network requests.

For local GGUF inference, use [`decompute-sdk`](https://crates.io/crates/decompute-sdk).
