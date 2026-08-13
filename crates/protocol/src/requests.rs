use crate::{ChatMessage, ChatRole, FinishReason, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub request_id: Uuid,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ChatMessage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: usize,
}

impl GenerateRequest {
    pub fn normalized_messages(&self) -> Result<Vec<ChatMessage>, RequestValidationError> {
        normalize_messages(self.prompt.as_deref(), self.messages.as_deref())
    }

    pub fn validate_tools(&self) -> Result<(), RequestValidationError> {
        validate_tools(&self.tools)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub request_id: Uuid,
    pub worker_id: String,
    pub model: String,
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RequestValidationError {
    #[error("provide exactly one of prompt or messages")]
    ExactlyOneOfPromptOrMessages,
    #[error("messages must not be empty")]
    EmptyMessages,
    #[error("tool name must use ASCII letters, digits, underscores, or hyphens")]
    InvalidToolName,
    #[error("tool names must be unique")]
    DuplicateToolName,
    #[error("tool parameters must be a JSON object")]
    InvalidToolParameters,
}

pub fn validate_tools(tools: &[ToolDefinition]) -> Result<(), RequestValidationError> {
    let mut names = std::collections::HashSet::new();
    for tool in tools {
        let name = &tool.function.name;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(RequestValidationError::InvalidToolName);
        }
        if !names.insert(name) {
            return Err(RequestValidationError::DuplicateToolName);
        }
        if !tool.function.parameters.is_object() {
            return Err(RequestValidationError::InvalidToolParameters);
        }
    }
    Ok(())
}

fn normalize_messages(
    prompt: Option<&str>,
    messages: Option<&[ChatMessage]>,
) -> Result<Vec<ChatMessage>, RequestValidationError> {
    match (prompt, messages) {
        (Some(prompt), None) => Ok(vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.to_owned(),
            tool_calls: vec![],
            tool_call_id: None,
        }]),
        (None, Some(messages)) if !messages.is_empty() => Ok(messages.to_vec()),
        (None, Some(_)) => Err(RequestValidationError::EmptyMessages),
        _ => Err(RequestValidationError::ExactlyOneOfPromptOrMessages),
    }
}

/// Private worker-to-coordinator streaming events. These are never exposed as
/// the public OpenAI-compatible wire format.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenerationStreamEvent {
    TextDelta {
        text: String,
    },
    Completed {
        input_tokens: usize,
        output_tokens: usize,
        tool_calls: Vec<ToolCall>,
        finish_reason: FinishReason,
    },
    Error {
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FunctionDefinition, ToolType};

    #[test]
    fn prompt_normalizes_to_one_user_message() {
        assert_eq!(
            normalize_messages(Some("hello"), None).unwrap(),
            vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".into(),
                tool_calls: vec![],
                tool_call_id: None,
            }]
        );
    }

    #[test]
    fn requires_exactly_one_input_shape() {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
            tool_calls: vec![],
            tool_call_id: None,
        }];
        assert_eq!(
            normalize_messages(None, None).unwrap_err(),
            RequestValidationError::ExactlyOneOfPromptOrMessages
        );
        assert_eq!(
            normalize_messages(Some("hello"), Some(&messages)).unwrap_err(),
            RequestValidationError::ExactlyOneOfPromptOrMessages
        );
        assert_eq!(
            normalize_messages(None, Some(&[])).unwrap_err(),
            RequestValidationError::EmptyMessages
        );
    }

    #[test]
    fn validates_tool_definitions() {
        let tool = ToolDefinition {
            kind: ToolType::Function,
            function: FunctionDefinition {
                name: "get-time".into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
            },
        };
        assert!(validate_tools(&[tool.clone()]).is_ok());
        assert_eq!(
            validate_tools(&[tool.clone(), tool]).unwrap_err(),
            RequestValidationError::DuplicateToolName
        );
    }
}
