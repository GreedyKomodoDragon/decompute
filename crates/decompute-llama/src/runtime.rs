use anyhow::{Context, Result, bail};
use decompute_core::{ChatMessage, ChatRole};
use encoding_rs::UTF_8;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static BACKEND_INIT: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug)]
pub struct GgufLoadConfig {
    pub context_tokens: u32,
    /// `None` keeps all layers on CPU. Set `Some(u32::MAX)` to offload all
    /// layers when the crate is compiled with an accelerator feature.
    pub gpu_layers: Option<u32>,
}

impl Default for GgufLoadConfig {
    fn default() -> Self {
        Self {
            context_tokens: 2_048,
            gpu_layers: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GgufGenerationConfig {
    pub max_tokens: usize,
    pub temperature: Option<f32>,
}

impl Default for GgufGenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufGenerationResult {
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// A loaded GGUF text model. The SDK owns it on a dedicated thread so this
/// synchronous native runtime never blocks an application's async executor.
pub struct GgufModel {
    model: LlamaModel,
    path: PathBuf,
    context_tokens: u32,
}

impl GgufModel {
    pub fn load(path: impl AsRef<Path>, config: GgufLoadConfig) -> Result<Self> {
        if config.context_tokens == 0 {
            bail!("GGUF context_tokens must be greater than zero");
        }
        let path = path.as_ref();
        super::validate_source(path)?;
        let mut parameters = LlamaModelParams::default();
        if let Some(layers) = config.gpu_layers {
            parameters = parameters.with_n_gpu_layers(layers);
        }
        let model = LlamaModel::load_from_file(backend()?, path, &parameters)
            .with_context(|| format!("load GGUF model {}", path.display()))?;
        Ok(Self {
            model,
            path: path.to_path_buf(),
            context_tokens: config.context_tokens,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Uses the template embedded in the GGUF rather than a hard-coded model family template.
    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let template = self
            .model
            .chat_template(None)
            .context("read embedded GGUF chat template")?;
        let messages = messages
            .iter()
            .map(|message| {
                LlamaChatMessage::new(
                    match message.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                        ChatRole::Tool => "tool",
                    }
                    .to_owned(),
                    message.content.clone(),
                )
                .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        self.model
            .apply_chat_template(&template, &messages, true)
            .context("apply embedded GGUF chat template")
    }

    pub fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        config: GgufGenerationConfig,
        callback: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<GgufGenerationResult> {
        let prompt = self.render_chat(messages)?;
        self.generate(&prompt, config, callback)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        config: GgufGenerationConfig,
        callback: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<GgufGenerationResult> {
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .context("tokenize GGUF prompt")?;
        if tokens.is_empty() {
            bail!("GGUF prompt tokenized to zero tokens");
        }
        let maximum = self.context_tokens as usize;
        if tokens.len() >= maximum {
            bail!(
                "GGUF prompt has {} tokens but context capacity is {}",
                tokens.len(),
                maximum
            );
        }
        let context_params =
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.context_tokens));
        let mut context = self
            .model
            .new_context(backend()?, context_params)
            .context("create GGUF inference context")?;
        let mut batch = LlamaBatch::new(maximum, 1);
        batch
            .add_sequence(&tokens, 0, false)
            .context("queue GGUF prompt tokens")?;
        context.decode(&mut batch).context("prefill GGUF prompt")?;

        let mut sampler = match config.temperature {
            Some(temperature) if temperature > 0.0 => {
                LlamaSampler::chain_simple([LlamaSampler::temp(temperature), LlamaSampler::dist(0)])
            }
            _ => LlamaSampler::greedy(),
        };
        sampler.accept_many(&tokens);
        let mut output = String::new();
        let mut decoder = UTF_8.new_decoder();
        let mut position = tokens.len();

        for _ in 0..config.max_tokens {
            let token = sampler.sample(&context, -1);
            if self.model.is_eog_token(token) {
                break;
            }
            sampler.accept(token);
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .context("decode GGUF output token")?;
            output.push_str(&piece);
            if !piece.is_empty() {
                callback(&piece)?;
            }
            if position >= maximum {
                break;
            }
            batch.clear();
            batch
                .add(
                    token,
                    i32::try_from(position).context("GGUF token position overflow")?,
                    &[0],
                    true,
                )
                .context("queue generated GGUF token")?;
            context
                .decode(&mut batch)
                .context("decode generated GGUF token")?;
            position += 1;
        }
        Ok(GgufGenerationResult {
            text: output,
            input_tokens: tokens.len(),
            output_tokens: position - tokens.len(),
        })
    }
}

fn backend() -> Result<&'static LlamaBackend> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let _guard = BACKEND_INIT
        .lock()
        .map_err(|_| anyhow::anyhow!("llama.cpp backend initialization lock poisoned"))?;
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let backend = LlamaBackend::init().context("initialize llama.cpp backend")?;
    BACKEND
        .set(backend)
        .map_err(|_| anyhow::anyhow!("llama.cpp backend initialized concurrently"))?;
    BACKEND
        .get()
        .ok_or_else(|| anyhow::anyhow!("llama.cpp backend did not initialize"))
}
