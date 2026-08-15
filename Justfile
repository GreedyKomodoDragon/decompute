set shell := ["zsh", "-cu"]

model_path := env_var_or_default("MODEL", "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
llvm_prefix := `brew --prefix llvm`

default:
    @just --list

# Download the default Qwen GGUF used by the worker recipes.
download-model:
    hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models

# Start the coordinator at http://127.0.0.1:8000.
coordinator:
    cargo run -p coordinator

# Start the first Metal-enabled local worker at http://127.0.0.1:9001.
worker-a:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo run -p worker --features metal -- --port 9001 --node-id worker-a --coordinator http://127.0.0.1:8000 --model "{{model_path}}"

# Start the second Metal-enabled local worker at http://127.0.0.1:9002.
worker-b:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo run -p worker --features metal -- --port 9002 --node-id worker-b --coordinator http://127.0.0.1:8000 --model "{{model_path}}"

# Run the full test suite using Homebrew LLVM for llama.cpp's native build.
test:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo test --workspace

# Compile the Metal worker without starting services.
check:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo check -p worker --features metal
