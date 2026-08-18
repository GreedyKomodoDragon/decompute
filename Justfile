set shell := ["zsh", "-cu"]

model_path := env_var_or_default("MODEL", "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
model_repo := "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
model_file := "qwen2.5-0.5b-instruct-q4_k_m.gguf"
# Pinned Hugging Face revision for the default model. Update this value and the
# digest together when deliberately upgrading the development model.
model_revision := "872f8a96064a1242ac3a3359cad77c3042548405"
model_sha256 := "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
llvm_prefix := `brew --prefix llvm`

default:
    @just --list

# Download the default Qwen GGUF used by the worker recipes.
download-model:
    hf download {{model_repo}} {{model_file}} --revision {{model_revision}} --local-dir models
    shasum -a 256 models/{{model_file}} | grep {{model_sha256}}

# Start the coordinator at http://127.0.0.1:8000.
coordinator:
    cargo run -p coordinator

# Start the first Metal-enabled local worker at http://127.0.0.1:9001.
worker-a:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo run -p worker --features metal -- --device metal --port 9001 --node-id worker-a --coordinator http://127.0.0.1:8000 --model "{{model_path}}"

# Start the second Metal-enabled local worker at http://127.0.0.1:9002.
worker-b:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo run -p worker --features metal -- --device metal --port 9002 --node-id worker-b --coordinator http://127.0.0.1:8000 --model "{{model_path}}"

# Run the full test suite using Homebrew LLVM for llama.cpp's native build.
test:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo test --workspace

# Compile the Metal worker without starting services.
check:
    CC="{{llvm_prefix}}/bin/clang" CXX="{{llvm_prefix}}/bin/clang++" cargo check -p worker --features metal

# Start the native macOS OpenAI-compatible chat harness.
harness:
    cargo run -p harness
