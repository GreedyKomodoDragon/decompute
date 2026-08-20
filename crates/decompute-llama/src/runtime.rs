use anyhow::{Context, Result, bail};
use decompute_core::{
    ChatMessage, ChatRole, FinishReason, SessionCacheConfig, SessionCacheMiss, SessionCacheStats,
    ToolCall, ToolDefinition,
};
use encoding_rs::UTF_8;
use llama_cpp_2::{
    context::LlamaContext,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::{
    collections::HashMap,
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static BACKEND_INIT: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug)]
pub struct GgufLoadConfig {
    pub context_tokens: u32,
    /// `None` keeps all layers on CPU. Set `Some(u32::MAX)` to offload all
    /// layers when the crate is compiled with an accelerator feature.
    pub gpu_layers: Option<u32>,
    pub session_cache: SessionCacheConfig,
    pub session_cache_stats: Option<Arc<SessionCacheStats>>,
}

impl Default for GgufLoadConfig {
    fn default() -> Self {
        Self {
            context_tokens: 2_048,
            gpu_layers: None,
            session_cache: SessionCacheConfig::default(),
            session_cache_stats: None,
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

#[derive(Clone, Debug, PartialEq)]
pub struct GgufGenerationResult {
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
}

/// A loaded GGUF text model. The SDK owns it on a dedicated thread so this
/// synchronous native runtime never blocks an application's async executor.
pub struct GgufModel {
    model: LlamaModel,
    path: PathBuf,
    context_tokens: u32,
    templates: super::TemplateRegistry,
    qwen_tool_calls: bool,
    session_cache_config: SessionCacheConfig,
    session_cache_stats: Option<Arc<SessionCacheStats>>,
    session_cache: HashMap<Uuid, SessionCacheEntry>,
}

struct SessionCacheEntry {
    context: LlamaContext<'static>,
    tokens: Vec<llama_cpp_2::token::LlamaToken>,
    estimated_bytes: usize,
    last_used: Instant,
}

// llama.cpp contexts are deliberately confined to the SDK's single model
// thread. They are never concurrently accessed; this marker only permits the
// already-created model actor to move its owned state into that thread.
unsafe impl Send for SessionCacheEntry {}

impl GgufModel {
    pub fn load(path: impl AsRef<Path>, config: GgufLoadConfig) -> Result<Self> {
        if config.context_tokens == 0 {
            bail!("GGUF context_tokens must be greater than zero");
        }
        let path = path.as_ref();
        super::validate_source(path)?;
        let info = super::inspect(path)?;
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
            templates: super::TemplateRegistry::load(path, &info.architecture)?,
            qwen_tool_calls: info.architecture.to_ascii_lowercase().starts_with("qwen"),
            session_cache_config: config.session_cache,
            session_cache_stats: config.session_cache_stats,
            session_cache: HashMap::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Uses the GGUF embedded template unless a named override or tool-aware
    /// template is required.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        requested_template: Option<&str>,
    ) -> Result<String> {
        if let Some(template) = self
            .templates
            .select(requested_template, !tools.is_empty())?
        {
            return self.templates.render(&template, messages, tools);
        }
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

    // The explicit inputs keep the runtime boundary clear: callers provide
    // normalized chat, template selection, generation controls, cancellation,
    // and streaming independently rather than constructing a runtime-specific
    // request type.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        requested_template: Option<&str>,
        session_id: Option<Uuid>,
        request_id: Uuid,
        cancellation: &CancellationToken,
        config: GgufGenerationConfig,
        callback: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<GgufGenerationResult> {
        if !tools.is_empty() && !self.qwen_tool_calls {
            bail!("tool-call parsing is currently implemented for Qwen GGUF models only");
        }
        let prompt = self.render_chat(messages, tools, requested_template)?;
        // Tool markup is only parsed after completion, so it must not leak as
        // visible OpenAI text in streamed responses.
        let mut deferred = String::new();
        let mut capture = |piece: &str| {
            if tools.is_empty() {
                callback(piece)
            } else {
                deferred.push_str(piece);
                Ok(())
            }
        };
        let mut generated =
            self.generate(&prompt, session_id, cancellation, config, &mut capture)?;
        if !tools.is_empty() {
            let (text, tool_calls) = match super::parse_tool_calls(&deferred, tools, request_id) {
                Ok(result) => result,
                Err(error) => {
                    self.invalidate_session(session_id);
                    return Err(error);
                }
            };
            if !text.is_empty() {
                callback(&text)?;
            }
            generated.text = text;
            generated.finish_reason = if tool_calls.is_empty() {
                FinishReason::Stop
            } else {
                FinishReason::ToolCalls
            };
            generated.tool_calls = tool_calls;
        }
        Ok(generated)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        session_id: Option<Uuid>,
        cancellation: &CancellationToken,
        config: GgufGenerationConfig,
        callback: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<GgufGenerationResult> {
        check_cancelled(cancellation)?;
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .context("tokenize GGUF prompt")?;
        if tokens.is_empty() {
            bail!("GGUF prompt tokenized to zero tokens");
        }
        let maximum = self.context_tokens as usize;
        let cache_enabled = session_id.is_some() && self.session_cache_config.max_entries > 0;
        if !cache_enabled {
            self.cache_miss();
            tracing::debug!(cache = "miss", reason = %SessionCacheMiss::Disabled.as_str(), "session cache bypassed");
        } else if tokens.len() < self.session_cache_config.min_tokens {
            self.cache_miss();
            tracing::debug!(cache = "miss", reason = %SessionCacheMiss::TooShort.as_str(), "session cache bypassed for short prompt");
        }
        if tokens.len() >= maximum {
            bail!(
                "GGUF prompt has {} tokens but context capacity is {}",
                tokens.len(),
                maximum
            );
        }
        let context_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.context_tokens))
            .with_n_seq_max(3);
        let now = Instant::now();
        let stats = self.session_cache_stats.clone();
        self.session_cache.retain(|_, entry| {
            if self.session_cache_config.idle_ttl_ms > 0
                && now.duration_since(entry.last_used) > self.session_cache_config.idle_ttl()
            {
                entry.context.clear_kv_cache();
                if let Some(stats) = &stats {
                    stats.increment_expirations();
                }
                false
            } else {
                true
            }
        });
        let cached = session_id
            .filter(|_| cache_enabled && tokens.len() >= self.session_cache_config.min_tokens)
            .and_then(|id| self.session_cache.remove(&id).map(|entry| (id, entry)));
        let (mut context, prefix) = match cached {
            Some((_, mut entry)) if tokens.starts_with(&entry.tokens) => {
                self.cache_hit();
                tracing::debug!(
                    cache = "hit",
                    reused_tokens = entry.tokens.len(),
                    "reusing worker-local inference session"
                );
                let prefix = entry.tokens.len();
                entry.last_used = now;
                (entry.context, prefix)
            }
            Some((_, mut entry)) => {
                self.cache_miss();
                entry.context.clear_kv_cache();
                tracing::debug!(
                    cache = "miss",
                    reason = %SessionCacheMiss::PrefixMismatch.as_str(),
                    "discarding worker-local inference session"
                );
                (
                    self.model
                        .new_context(backend()?, context_params)
                        .context("create GGUF inference context")?,
                    0,
                )
            }
            None => {
                self.cache_miss();
                (
                    self.model
                        .new_context(backend()?, context_params)
                        .context("create GGUF inference context")?,
                    0,
                )
            }
        };
        context
            .clear_kv_cache_seq(Some(1), None, None)
            .context("clear GGUF generation sequence")?;
        if prefix > 0 {
            context
                .copy_kv_cache_seq(0, 1, None, None)
                .context("copy GGUF committed checkpoint")?;
        }
        let mut batch = LlamaBatch::new(maximum, 1);
        batch
            .add_sequence(&tokens[prefix..], 1, false)
            .context("queue GGUF prompt tokens")?;
        if prefix < tokens.len() {
            context.decode(&mut batch).context("prefill GGUF prompt")?;
        }
        check_cancelled(cancellation)?;

        let checkpoint_tokens = tokens.clone();
        // Sequence zero is the last committed prompt checkpoint. Sequence
        // one is the private generation copy and sequence two is the candidate
        // checkpoint. This makes cache publication transactional: failures
        // leave sequence zero untouched.
        context
            .clear_kv_cache_seq(Some(2), None, None)
            .context("clear GGUF candidate checkpoint")?;
        context
            .copy_kv_cache_seq(1, 2, None, None)
            .context("copy GGUF prompt checkpoint")?;
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
            check_cancelled(cancellation)?;
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
            check_cancelled(cancellation)?;
            if position >= maximum {
                break;
            }
            batch.clear();
            batch
                .add(
                    token,
                    i32::try_from(position).context("GGUF token position overflow")?,
                    &[1],
                    true,
                )
                .context("queue generated GGUF token")?;
            context
                .decode(&mut batch)
                .context("decode generated GGUF token")?;
            position += 1;
        }
        if let Some(session_id) = session_id
            .filter(|_| cache_enabled && tokens.len() >= self.session_cache_config.min_tokens)
        {
            let estimated_bytes =
                checkpoint_tokens.len() * std::mem::size_of::<llama_cpp_2::token::LlamaToken>();
            if self.session_cache_config.max_bytes > 0
                && estimated_bytes > self.session_cache_config.max_bytes
            {
                tracing::debug!(cache = "miss", reason = %SessionCacheMiss::Overflow.as_str(), "session cache entry exceeds byte budget");
                return Ok(GgufGenerationResult {
                    text: output,
                    input_tokens: tokens.len(),
                    output_tokens: position - tokens.len(),
                    tool_calls: vec![],
                    finish_reason: FinishReason::Stop,
                });
            }
            while self.session_cache.len() >= self.session_cache_config.max_entries
                || (self.session_cache_config.max_bytes > 0
                    && self
                        .session_cache
                        .values()
                        .map(|entry| entry.estimated_bytes)
                        .sum::<usize>()
                        + estimated_bytes
                        > self.session_cache_config.max_bytes)
            {
                if let Some(evicted) = self
                    .session_cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(id, _)| *id)
                {
                    if let Some(mut entry) = self.session_cache.remove(&evicted) {
                        entry.context.clear_kv_cache();
                    }
                    tracing::debug!(
                        cache = "eviction",
                        reason = "lru_or_byte_budget",
                        "evicted least-recently-used worker session"
                    );
                    self.cache_eviction();
                } else {
                    break;
                }
            }
            // Contexts are dropped before the owning model. The actor thread
            // is the sole owner, so extending this borrow is sound here.
            let mut context =
                unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(context) };
            // Publish only after the complete generation succeeded.
            context
                .clear_kv_cache_seq(Some(0), None, None)
                .context("clear GGUF committed checkpoint")?;
            context
                .copy_kv_cache_seq(2, 0, None, None)
                .context("commit GGUF prompt checkpoint")?;
            self.session_cache.insert(
                session_id,
                SessionCacheEntry {
                    context,
                    tokens: checkpoint_tokens,
                    estimated_bytes,
                    last_used: now,
                },
            );
        }
        Ok(GgufGenerationResult {
            text: output,
            input_tokens: tokens.len(),
            output_tokens: position - tokens.len(),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
        })
    }

    fn invalidate_session(&mut self, session_id: Option<Uuid>) {
        if let Some(session_id) = session_id
            && let Some(mut entry) = self.session_cache.remove(&session_id)
        {
            entry.context.clear_kv_cache();
            self.cache_invalidation();
        }
    }

    fn cache_hit(&self) {
        if let Some(stats) = &self.session_cache_stats {
            stats.increment_hits();
        }
    }

    fn cache_miss(&self) {
        if let Some(stats) = &self.session_cache_stats {
            stats.increment_misses();
        }
    }

    fn cache_eviction(&self) {
        if let Some(stats) = &self.session_cache_stats {
            stats.increment_evictions();
        }
    }

    fn cache_invalidation(&self) {
        if let Some(stats) = &self.session_cache_stats {
            stats.increment_invalidations();
        }
    }
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        bail!("generation cancelled");
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::check_cancelled;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn cancelled_token_stops_generation_at_the_next_safe_boundary() {
        let cancellation = CancellationToken::new();
        assert!(check_cancelled(&cancellation).is_ok());
        cancellation.cancel();
        assert!(check_cancelled(&cancellation).is_err());
    }
}
