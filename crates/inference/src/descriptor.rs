use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    pub directory: PathBuf,
    pub model_type: String,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: PathBuf,
    pub weight_paths: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct ConfigHeader {
    model_type: String,
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
        Ok(Self {
            directory,
            model_type: config.model_type,
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            weight_paths: vec![weights_path],
        })
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
        assert_eq!(
            ModelDescriptor::load(directory.path()).unwrap().model_type,
            "qwen2"
        );
    }
}
