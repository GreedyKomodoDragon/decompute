# Contributing to Decompute

Thanks for improving Decompute. It is an experimental macOS/Metal local-inference prototype; small, focused changes are preferred over broad production features.

## Development setup

1. Install Rust using the pinned toolchain in `rust-toolchain.toml`.
2. Install Homebrew LLVM and `just`:

   ```bash
   brew install llvm just lefthook
   ```

3. Install the repository hooks once:

   ```bash
   lefthook install
   ```

   Pre-commit runs formatting, Clippy, and `cargo check` across all workspace targets and features. Pre-push runs the full test suite.

4. Run the full validation suite:

   ```bash
   just test
   cargo fmt --check
   CC="$(brew --prefix llvm)/bin/clang" \
   CXX="$(brew --prefix llvm)/bin/clang++" \
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   ```

The native llama.cpp dependency must use Homebrew LLVM on macOS. Tests do not require downloading a model.

## Pull requests

- Keep a pull request focused on one concern.
- Add or update tests for behavior changes.
- Update the README when setup, supported behavior, or public API changes.
- Do not commit GGUF model files, build output, credentials, or local configuration.
- Preserve the boundary: coordinators and protocol types must not depend on model runtimes.

## Scope

Please open an issue before undertaking a large runtime, provider, or protocol redesign. The current target is macOS on Apple Silicon; authentication, payments, P2P transport, and multi-platform support are intentionally out of scope for now.
