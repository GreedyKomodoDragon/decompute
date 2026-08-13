use crate::execution::ModelPrecision;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    pub directory: PathBuf,
    pub model_type: String,
    pub declared_precision: Option<ModelPrecision>,
    pub stored_precision: ModelPrecision,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: PathBuf,
    pub weight_paths: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct ConfigHeader {
    model_type: String,
    #[serde(default)]
    torch_dtype: Option<String>,
}

impl ModelDescriptor {
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();
        let config_path = directory.join("config.json");
        let tokenizer_path = directory.join("tokenizer.json");
        let tokenizer_config_path = directory.join("tokenizer_config.json");
        let weights_path = directory.join("model.safetensors");
        for path in [
            &config_path,
            &tokenizer_path,
            &tokenizer_config_path,
            &weights_path,
        ] {
            if !path.exists() {
                bail!("missing model file {}", path.display());
            }
        }
        let config: ConfigHeader = serde_json::from_reader(fs::File::open(&config_path)?)
            .with_context(|| format!("parse {}", config_path.display()))?;
        let declared_precision = config
            .torch_dtype
            .as_deref()
            .map(ModelPrecision::parse)
            .transpose()
            .with_context(|| format!("parse torch_dtype in {}", config_path.display()))?;
        let stored_precision = safetensors_precision(&weights_path)?;
        if let Some(declared) = declared_precision {
            if declared != stored_precision {
                bail!(
                    "model config declares {declared} but safetensors weights are {stored_precision}"
                );
            }
        }
        Ok(Self {
            directory,
            model_type: config.model_type,
            declared_precision,
            stored_precision,
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            weight_paths: vec![weights_path],
        })
    }
}

fn safetensors_precision(path: &Path) -> Result<ModelPrecision> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)
        .context("read safetensors header length")?;
    let header_length = u64::from_le_bytes(length);
    const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
    if header_length > MAX_HEADER_BYTES {
        bail!("safetensors header is too large: {header_length} bytes");
    }
    let mut header = vec![0; header_length as usize];
    file.read_exact(&mut header)
        .context("read safetensors header")?;
    let header: serde_json::Value =
        serde_json::from_slice(&header).context("parse safetensors header")?;
    let dtypes = header
        .as_object()
        .context("safetensors header is not an object")?
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
        .filter_map(|(_, tensor)| tensor.get("dtype")?.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    match dtypes.len() {
        0 => bail!("safetensors header contains no tensor dtypes"),
        1 => ModelPrecision::parse(dtypes.into_iter().next().expect("length checked")),
        _ => bail!(
            "mixed safetensors dtypes are not supported: {}",
            dtypes.into_iter().collect::<Vec<_>>().join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_model_type() {
        let directory = tempfile::tempdir().unwrap();
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ] {
            fs::write(directory.path().join(name), "").unwrap();
        }
        fs::write(
            directory.path().join("config.json"),
            r#"{"model_type":"qwen2"}"#,
        )
        .unwrap();
        // A minimal safetensors header with a single F32 tensor.
        let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        fs::write(
            directory.path().join("model.safetensors"),
            [header.len().to_le_bytes().as_slice(), header.as_slice()].concat(),
        )
        .unwrap();
        let descriptor = ModelDescriptor::load(directory.path()).unwrap();
        assert_eq!(descriptor.model_type, "qwen2");
        assert_eq!(descriptor.stored_precision, ModelPrecision::F32);
    }

    #[test]
    fn rejects_declared_precision_that_disagrees_with_weights() {
        let directory = tempfile::tempdir().unwrap();
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            fs::write(directory.path().join(name), "").unwrap();
        }
        fs::write(
            directory.path().join("config.json"),
            r#"{"model_type":"qwen2","torch_dtype":"bfloat16"}"#,
        )
        .unwrap();
        let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        fs::write(
            directory.path().join("model.safetensors"),
            [header.len().to_le_bytes().as_slice(), header.as_slice()].concat(),
        )
        .unwrap();
        assert!(
            ModelDescriptor::load(directory.path())
                .unwrap_err()
                .to_string()
                .contains("config declares bf16 but safetensors weights are f32")
        );
    }
}
