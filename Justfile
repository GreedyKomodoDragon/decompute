set shell := ["zsh", "-cu"]

worker_a_model_id := env_var_or_default("WORKER_A_MODEL_ID", "qwen2.5-0.5b-instruct-q4-k-m")
worker_b_model_id := env_var_or_default("WORKER_B_MODEL_ID", "qwen2.5-0.5b-instruct-q4-k-m")
llvm_prefix := `brew --prefix llvm`

default:
    @just --list

# Start the coordinator at http://127.0.0.1:8000.
coordinator:
    cargo run -p coordinator

# Start the first Metal-enabled local worker at http://127.0.0.1:9001.
worker-a:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo run -p worker --features metal -- --device metal --port 9001 --node-id worker-a --coordinator http://127.0.0.1:8000 --model "{{worker_a_model_id}}"

# Start the second Metal-enabled local worker at http://127.0.0.1:9002.
worker-b:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo run -p worker --features metal -- --device metal --port 9002 --node-id worker-b --coordinator http://127.0.0.1:8000 --model "{{worker_b_model_id}}"

# Start worker B with the larger curated model for multi-model testing.
worker-b-1-5b:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo run -p worker --features metal -- --device metal --port 9002 --node-id worker-b --coordinator http://127.0.0.1:8000 --model qwen2.5-1.5b-instruct-q4-k-m

# Run the full test suite using Homebrew LLVM for llama.cpp's native build.
test:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo test --workspace

# Compile the Metal worker without starting services.
check:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo check -p worker --features metal

# Start the native macOS OpenAI-compatible chat harness.
harness:
    cargo run -p harness
