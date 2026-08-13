use decompute_core::{FinishReason, ToolCall};

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

impl From<inference::GenerationResult> for GenerationResult {
    fn from(value: inference::GenerationResult) -> Self {
        Self {
            text: value.text,
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            tool_calls: value.tool_calls,
            finish_reason: value.finish_reason,
        }
    }
}

impl From<GenerationConfig> for inference::GenerationConfig {
    fn from(value: GenerationConfig) -> Self {
        Self {
            max_tokens: value.max_tokens,
            temperature: value.temperature,
        }
    }
}
