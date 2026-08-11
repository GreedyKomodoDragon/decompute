use crate::{GenerationConfig, GenerationResult, TokenCallback, tokenizer};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::qwen2};
use protocol::{Acceleration, HardwareInfo, ModelFile, ModelManifest};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use sysinfo::System;
use tokenizers::Tokenizer;

pub struct LocalModel {
    tokenizer: Tokenizer,
    model: qwen2::ModelForCausalLM,
    device: Device,
    end_tokens: Vec<u32>,
}

impl LocalModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_on_device(path, Device::Cpu)
    }

    pub fn load_on_device(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = path.as_ref();
        let config_path = path.join("config.json");
        let tokenizer_path = path.join("tokenizer.json");
        let weights_path = path.join("model.safetensors");
        for required in [&config_path, &tokenizer_path, &weights_path] {
            if !required.exists() {
                bail!("missing model file {}", required.display());
            }
        }
        let config: qwen2::Config = serde_json::from_reader(fs::File::open(&config_path)?)
            .context("parse Qwen config.json")?;
        let tokenizer = tokenizer::load(&tokenizer_path)?;
        let dtype = execution_dtype(&weights_path, &device)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device) }
            .context("memory-map model.safetensors")?;
        let model = qwen2::ModelForCausalLM::new(&config, vb).context("load Qwen model")?;
        let end_tokens = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|token| tokenizer.token_to_id(token))
            .collect();
        Ok(Self {
            tokenizer,
            model,
            device,
            end_tokens,
        })
    }

    pub fn generate(&mut self, prompt: &str, config: GenerationConfig) -> Result<GenerationResult> {
        self.generate_with_callback(prompt, config, &mut |_| Ok(()))
    }

    pub fn generate_with_callback(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        callback: &mut TokenCallback<'_>,
    ) -> Result<GenerationResult> {
        let prompt = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!(err))?;
        let mut tokens = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            bail!("prompt tokenized to zero tokens");
        }
        let input_tokens = tokens.len();
        self.model.clear_kv_cache();
        let mut logits_processor = LogitsProcessor::new(299_792_458, config.temperature, None);
        let mut output = String::new();
        let result = (|| {
            for index in 0..config.max_tokens {
                let context_size = if index == 0 { tokens.len() } else { 1 };
                let input = Tensor::new(&tokens[tokens.len() - context_size..], &self.device)
                    .context("create input token tensor")?
                    .unsqueeze(0)
                    .context("add batch dimension")?;
                let logits = self
                    .model
                    .forward(&input, tokens.len() - context_size)
                    .context("Qwen forward pass")?
                    .squeeze(0)
                    .context("remove Qwen batch dimension")?
                    .squeeze(0)
                    .context("remove Qwen sequence dimension")?;
                let token = logits_processor
                    .sample(&logits)
                    .context("sample next token")?;
                if self.end_tokens.contains(&token) {
                    break;
                }
                tokens.push(token);
                let piece = self
                    .tokenizer
                    .decode(&[token], false)
                    .map_err(|err| anyhow::anyhow!(err))?;
                callback(&piece)?;
                output.push_str(&piece);
            }
            Ok(GenerationResult {
                text: output,
                input_tokens,
                output_tokens: tokens.len() - input_tokens,
            })
        })();
        self.model.clear_kv_cache();
        result
    }
}

fn execution_dtype(weights_path: &Path, device: &Device) -> Result<DType> {
    let source_dtype = safetensors_dtype(weights_path)?;
    execution_dtype_for(device.is_cpu(), &source_dtype)
}

fn execution_dtype_for(cpu: bool, source_dtype: &str) -> Result<DType> {
    match (cpu, source_dtype) {
        // Candle's CPU matmul supports F32, but not BF16. Promote only when the
        // model's safetensors header says the stored weights need it.
        (true, "F32") => Ok(DType::F32),
        (true, "BF16" | "F16") => Ok(DType::F32),
        (false, "F32") => Ok(DType::F32),
        (false, "BF16") => Ok(DType::BF16),
        (false, "F16") => Ok(DType::F16),
        (_, dtype) => bail!("unsupported floating-point safetensors dtype {dtype}"),
    }
}

fn safetensors_dtype(path: &Path) -> Result<String> {
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
        1 => Ok(dtypes
            .into_iter()
            .next()
            .expect("length checked")
            .to_owned()),
        _ => bail!(
            "mixed safetensors dtypes are not supported: {}",
            dtypes.into_iter().collect::<Vec<_>>().join(", ")
        ),
    }
}

pub fn local_manifest(path: impl AsRef<Path>) -> Result<ModelManifest> {
    let root = path.as_ref();
    let names = ["config.json", "tokenizer.json", "model.safetensors"];
    let files = names
        .iter()
        .map(|name| {
            let path = root.join(name);
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            Ok(ModelFile {
                path: (*name).to_owned(),
                sha256: format!("{:x}", Sha256::digest(bytes)),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut hasher = Sha256::new();
    for file in &files {
        hasher.update(file.path.as_bytes());
        hasher.update(file.sha256.as_bytes());
    }
    Ok(ModelManifest {
        id: format!("sha256:{:x}", hasher.finalize()),
        architecture: "qwen2".into(),
        revision: "local".into(),
        quantization: None,
        files,
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

pub fn model_directory(path: impl AsRef<Path>) -> PathBuf {
    path.as_ref().to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_promotes_half_precision_weight_formats() {
        assert_eq!(execution_dtype_for(true, "BF16").unwrap(), DType::F32);
        assert_eq!(execution_dtype_for(true, "F16").unwrap(), DType::F32);
        assert_eq!(execution_dtype_for(true, "F32").unwrap(), DType::F32);
    }

    #[test]
    fn accelerators_preserve_supported_weight_formats() {
        assert_eq!(execution_dtype_for(false, "BF16").unwrap(), DType::BF16);
        assert_eq!(execution_dtype_for(false, "F16").unwrap(), DType::F16);
    }
}
