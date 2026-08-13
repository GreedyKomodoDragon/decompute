//! llama.cpp/GGUF runtime building blocks for the Decompute SDK.
//!
//! This crate deliberately owns native-runtime details. Networking crates do
//! not depend on it; the SDK will expose a stable actor API over this layer.

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

#[cfg(feature = "runtime")]
mod runtime;

#[cfg(feature = "runtime")]
use anyhow::Context;
#[cfg(feature = "runtime")]
use llama_cpp_2::gguf::GgufContext;

#[cfg(feature = "runtime")]
pub use runtime::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufModelInfo {
    pub path: PathBuf,
    pub architecture: String,
    pub name: Option<String>,
    pub chat_template: Option<String>,
    pub tensor_count: i64,
}

/// Reads GGUF metadata without loading model tensor data.
pub fn inspect(path: impl AsRef<Path>) -> Result<GgufModelInfo> {
    inspect_impl(path.as_ref())
}

#[cfg(feature = "runtime")]
fn inspect_impl(path: &Path) -> Result<GgufModelInfo> {
    let context = GgufContext::from_file(path)
        .with_context(|| format!("read GGUF metadata from {}", path.display()))?;
    let metadata = |key: &str| {
        let index = context.find_key(key);
        (index >= 0)
            .then(|| context.val_str(index))
            .flatten()
            .map(str::to_owned)
    };
    let architecture = metadata("general.architecture")
        .or_else(|| metadata("llama.architecture"))
        .ok_or_else(|| anyhow::anyhow!("GGUF model has no general.architecture metadata"))?;
    Ok(GgufModelInfo {
        path: path.to_path_buf(),
        architecture,
        name: metadata("general.name"),
        chat_template: metadata("tokenizer.chat_template"),
        tensor_count: context.n_tensors(),
    })
}

#[cfg(not(feature = "runtime"))]
fn inspect_impl(_path: &Path) -> Result<GgufModelInfo> {
    bail!("GGUF runtime is disabled; enable decompute-llama's `runtime` feature")
}

/// Validates a model source is a GGUF file before scheduling a native load.
pub fn validate_source(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        bail!("expected a .gguf model file, got {}", path.display());
    }
    inspect(path).map(|_| ())
}
