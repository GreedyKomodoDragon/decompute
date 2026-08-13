use crate::{
    GenerationConfig, GenerationResult, TokenCallback,
    descriptor::ModelDescriptor,
    execution::{ExecutionPlan, ExecutionTarget},
    prompt_template::TemplateBundle,
    provider::{CausalModelBackend, ProviderRegistry},
    tokenizer,
    tool_calls::ToolCallParser,
};
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use protocol::{Acceleration, ChatMessage, HardwareInfo, ModelFile, ModelManifest, ToolDefinition};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use sysinfo::System;
use tokenizers::Tokenizer;
use tracing::{info, warn};

pub struct LocalModel {
    tokenizer: Tokenizer,
    templates: TemplateBundle,
    backend: Box<dyn CausalModelBackend>,
    tool_call_parser: &'static dyn ToolCallParser,
    device: Device,
    end_tokens: Vec<u32>,
    execution_plan: ExecutionPlan,
}

impl LocalModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_on_device(path, Device::Cpu)
    }

    pub fn load_on_device(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let descriptor = ModelDescriptor::load(path)?;
        let target = execution_target(&device)?;
        let tokenizer = tokenizer::load(&descriptor.tokenizer_path)?;
        let templates = TemplateBundle::load(&descriptor)?;
        let registry = ProviderRegistry::builtin();
        let execution_plan = registry.execution_plan(&descriptor, target)?;
        let provider = registry.load(&descriptor, &device, execution_plan.dtype())?;
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
            execution_plan,
        })
    }

    pub fn execution_plan(&self) -> &ExecutionPlan {
        &self.execution_plan
    }

    /// Verify the actual backend/kernel path before advertising this model.
    pub fn smoke_test(&mut self) -> Result<()> {
        self.generate(
            &[ChatMessage {
                role: protocol::ChatRole::User,
                content: "ping".into(),
                tool_calls: vec![],
                tool_call_id: None,
            }],
            None,
            &[],
            uuid::Uuid::nil(),
            GenerationConfig {
                max_tokens: 1,
                temperature: None,
            },
        )
        .map(|_| ())
        .context("execution smoke test")
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
        let started = Instant::now();
        info!(
            request_id = %request_id,
            input_tokens,
            max_tokens = config.max_tokens,
            target = ?self.execution_plan.target,
            runtime_precision = ?self.execution_plan.runtime_precision,
            "starting local generation"
        );
        self.backend.clear_kv_cache();
        let mut logits_processor = LogitsProcessor::new(299_792_458, config.temperature, None);
        let mut output = String::new();
        let mut stream_filter = self.tool_call_parser.stream_filter();
        let mut first_visible_token = true;
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
                    info!(
                        request_id = %request_id,
                        output_tokens = tokens.len() - input_tokens,
                        elapsed_ms = started.elapsed().as_millis(),
                        "generation reached end token"
                    );
                    break;
                }
                tokens.push(token);
                let piece = self
                    .tokenizer
                    .decode(&[token], false)
                    .map_err(|err| anyhow::anyhow!(err))?;
                output.push_str(&piece);
                let visible = stream_filter.push(&piece);
                if !visible.is_empty() {
                    if first_visible_token {
                        info!(
                            request_id = %request_id,
                            elapsed_ms = started.elapsed().as_millis(),
                            "generated first visible token"
                        );
                        first_visible_token = false;
                    }
                    callback(&visible)?;
                }
            }
            let visible = stream_filter.finish();
            if !visible.is_empty() {
                callback(&visible)?;
            }
            let parsed = self.tool_call_parser.parse(&output, tools, request_id)?;
            let generated = GenerationResult {
                text: parsed.text,
                input_tokens,
                output_tokens: tokens.len() - input_tokens,
                tool_calls: parsed.tool_calls,
                finish_reason: parsed.finish_reason,
            };
            info!(
                request_id = %request_id,
                output_tokens = generated.output_tokens,
                tool_calls = generated.tool_calls.len(),
                finish_reason = ?generated.finish_reason,
                elapsed_ms = started.elapsed().as_millis(),
                "local generation completed"
            );
            Ok(generated)
        })();
        self.backend.clear_kv_cache();
        if let Err(err) = &result {
            warn!(
                request_id = %request_id,
                elapsed_ms = started.elapsed().as_millis(),
                error = %format!("{err:#}"),
                "local generation failed"
            );
        }
        result
    }
}

fn execution_target(device: &Device) -> Result<ExecutionTarget> {
    if device.is_cpu() {
        Ok(ExecutionTarget::Cpu)
    } else if device.is_metal() {
        Ok(ExecutionTarget::Metal)
    } else if device.is_cuda() {
        Ok(ExecutionTarget::Cuda)
    } else {
        bail!("unsupported Candle device")
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
    fn detects_cpu_execution_target() {
        assert_eq!(
            execution_target(&Device::Cpu).unwrap(),
            ExecutionTarget::Cpu
        );
    }
}
