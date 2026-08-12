use crate::{
    GenerationConfig, GenerationResult, TokenCallback,
    descriptor::ModelDescriptor,
    prompt_template::TemplateBundle,
    provider::{CausalModelBackend, ProviderRegistry},
    tokenizer,
    tool_calls::ToolCallParser,
};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use protocol::{Acceleration, ChatMessage, HardwareInfo, ModelFile, ModelManifest, ToolDefinition};
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
    templates: TemplateBundle,
    backend: Box<dyn CausalModelBackend>,
    tool_call_parser: &'static dyn ToolCallParser,
    device: Device,
    end_tokens: Vec<u32>,
}

impl LocalModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_on_device(path, Device::Cpu)
    }

    pub fn load_on_device(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let descriptor = ModelDescriptor::load(path)?;
        let tokenizer = tokenizer::load(&descriptor.tokenizer_path)?;
        let templates = TemplateBundle::load(&descriptor)?;
        let dtype = execution_dtype(&descriptor.weight_paths[0], &device)?;
        let provider = ProviderRegistry::builtin().load(&descriptor, &device, dtype)?;
        let mut end_tokens = templates
            .eos_token()
            .into_iter()
            .chain(["<|endoftext|>"])
            .filter_map(|token| tokenizer.token_to_id(token))
            .collect::<Vec<_>>();
        end_tokens.sort_unstable();
        end_tokens.dedup();
        Ok(Self {
            tokenizer,
            templates,
            backend: provider.backend,
            tool_call_parser: provider.tool_call_parser,
            device,
            end_tokens,
        })
    }

    pub fn generate(
        &mut self,
        messages: &[ChatMessage],
        template: Option<&str>,
        tools: &[ToolDefinition],
        request_id: uuid::Uuid,
        config: GenerationConfig,
    ) -> Result<GenerationResult> {
        self.generate_with_callback(messages, template, tools, request_id, config, &mut |_| {
            Ok(())
        })
    }

    pub fn generate_with_callback(
        &mut self,
        messages: &[ChatMessage],
        template: Option<&str>,
        tools: &[ToolDefinition],
        request_id: uuid::Uuid,
        config: GenerationConfig,
        callback: &mut TokenCallback<'_>,
    ) -> Result<GenerationResult> {
        let prompt = self.templates.render(messages, template, tools)?;
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!(err))?;
        let mut tokens = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            bail!("prompt tokenized to zero tokens");
        }
        let input_tokens = tokens.len();
        self.backend.clear_kv_cache();
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
                    .backend
                    .next_token_logits(&input, tokens.len() - context_size)?;
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
            let parsed = self.tool_call_parser.parse(&output, tools, request_id)?;
            Ok(GenerationResult {
                text: parsed.text,
                input_tokens,
                output_tokens: tokens.len() - input_tokens,
                tool_calls: parsed.tool_calls,
                finish_reason: parsed.finish_reason,
            })
        })();
        self.backend.clear_kv_cache();
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
    let descriptor = ModelDescriptor::load(path)?;
    let root = &descriptor.directory;
    let mut paths = vec![
        descriptor.config_path.clone(),
        descriptor.tokenizer_path.clone(),
        descriptor.tokenizer_config_path.clone(),
    ];
    paths.extend(descriptor.weight_paths.clone());
    let standalone_template = root.join("chat_template.jinja");
    if standalone_template.exists() {
        paths.push(standalone_template);
    }
    let files = paths
        .iter()
        .map(|name| {
            let path = name.clone();
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            Ok(ModelFile {
                path: path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
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
        architecture: descriptor.model_type,
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
