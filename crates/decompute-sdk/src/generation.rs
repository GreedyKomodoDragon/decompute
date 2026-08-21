use anyhow::Error;
use decompute_core::{ChatMessage, FinishReason, ToolCall, ToolDefinition};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub request_id: Uuid,
    pub session_id: Option<Uuid>,
    pub messages: Vec<ChatMessage>,
    /// Reserved for a future named-template registry. GGUF models currently
    /// use the template embedded in their own metadata.
    pub template: Option<String>,
    pub tools: Vec<ToolDefinition>,
    pub generation: GenerationConfig,
    /// Cooperative cancellation checked by the native model loop between safe
    /// llama.cpp execution steps.
    pub cancellation: CancellationToken,
}

#[derive(Debug)]
pub enum GenerationEvent {
    TextDelta(String),
    Completed(GenerationResult),
    Error(Error),
}

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: Option<f64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
}
