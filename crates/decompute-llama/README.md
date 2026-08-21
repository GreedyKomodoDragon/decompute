# decompute-llama

GGUF and llama.cpp runtime building blocks for Decompute.

The default feature set provides metadata, templates, and tool-call parsing.
Enable `runtime` for native inference or `metal` for Apple Silicon Metal
inference. Applications normally depend on [`decompute-sdk`](https://crates.io/crates/decompute-sdk)
instead of using this crate directly.

Native inference requires a compatible C++ toolchain and a local GGUF model.
