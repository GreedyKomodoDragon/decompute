//! llama.cpp/GGUF runtime building blocks for the Decompute SDK.
//!
//! This crate deliberately owns native-runtime details. Networking crates do
//! not depend on it; the SDK will expose a stable actor API over this layer.

use anyhow::{Context, Result, bail};
use decompute_core::{Acceleration, HardwareInfo, ModelFile, ModelManifest};
use sha2::{Digest, Sha256};
use std::{
    fs,
    num::NonZeroU32,
    path::{Path, PathBuf},
};
use sysinfo::System;

#[cfg(feature = "runtime")]
mod runtime;
mod templates;
mod tool_calls;

#[cfg(feature = "runtime")]
use llama_cpp_2::gguf::GgufContext;

#[cfg(feature = "runtime")]
pub use runtime::*;
pub use templates::TemplateRegistry;
pub use tool_calls::parse_tool_calls;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufModelInfo {
    pub path: PathBuf,
    pub architecture: String,
    /// Context capacity the model was trained to support, from GGUF metadata.
    pub trained_context_tokens: NonZeroU32,
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
    let context_key = format!("{architecture}.context_length");
    let context_index = context.find_key(&context_key);
    if context_index < 0 {
        bail!("GGUF model has no {context_key} metadata");
    }
    if context.kv_type(context_index) != llama_cpp_sys_2::GGUF_TYPE_UINT32 {
        bail!("GGUF metadata {context_key} must be an unsigned 32-bit integer");
    }
    let trained_context_tokens = NonZeroU32::new(context.val_u32(context_index))
        .ok_or_else(|| anyhow::anyhow!("GGUF metadata {context_key} must be greater than zero"))?;
    Ok(GgufModelInfo {
        path: path.to_path_buf(),
        architecture,
        trained_context_tokens,
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

/// Creates a deterministic local manifest for a single GGUF model file.
pub fn local_manifest(path: impl AsRef<Path>) -> Result<ModelManifest> {
    let path = path.as_ref();
    let info = inspect(path)?;
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let file = ModelFile {
        path: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    let mut hasher = Sha256::new();
    hasher.update(file.path.as_bytes());
    hasher.update(file.sha256.as_bytes());
    Ok(ModelManifest {
        id: format!("sha256:{:x}", hasher.finalize()),
        architecture: info.architecture,
        revision: "local".into(),
        quantization: None,
        files: vec![file],
    })
}

pub fn hardware_info(acceleration: Acceleration) -> HardwareInfo {
    let system = System::new_all();
    HardwareInfo {
        architecture: std::env::consts::ARCH.to_owned(),
        total_memory_bytes: system.total_memory(),
        available_memory_bytes: system.available_memory(),
        acceleration,
    }
}
