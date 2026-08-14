use protocol::{
    ChatMessage, ChatRole, FinishReason, FunctionCall, GenerateRequest, GenerateResponse, ToolCall,
    ToolDefinition, ToolType, validate_tools,
};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// Optional Decompute extension selecting a local named GGUF template.
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: OpenAiRole,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_calls: Vec<OpenAiToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RequestError {
    #[error("messages must not be empty")]
    EmptyMessages,
    #[error("message content must be text")]
    NonTextContent,
    #[error("tool call arguments for {0} must be a JSON object")]
    InvalidToolArguments(String),
    #[error(transparent)]
    Validation(#[from] protocol::RequestValidationError),
}

impl TryFrom<ChatCompletionRequest> for GenerateRequest {
    type Error = RequestError;

    fn try_from(request: ChatCompletionRequest) -> Result<Self, Self::Error> {
        if request.messages.is_empty() {
            return Err(RequestError::EmptyMessages);
        }
        validate_tools(&request.tools)?;
        let messages = request
            .messages
            .into_iter()
            .map(ChatMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GenerateRequest {
            request_id: Uuid::new_v4(),
            model: request.model,
            prompt: None,
            messages: Some(messages),
            template: request.template,
            tools: request.tools,
            max_tokens: request
                .max_completion_tokens
                .or(request.max_tokens)
                .unwrap_or(100),
        })
    }
}

impl TryFrom<ChatCompletionMessage> for ChatMessage {
    type Error = RequestError;

    fn try_from(message: ChatCompletionMessage) -> Result<Self, Self::Error> {
        let content = match message.content {
            None | Some(serde_json::Value::Null) => String::new(),
            Some(serde_json::Value::String(content)) => content,
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .map(|part| part.get("text").and_then(serde_json::Value::as_str))
                .collect::<Option<Vec<_>>>()
                .map(|parts| parts.concat())
                .ok_or(RequestError::NonTextContent)?,
            Some(_) => return Err(RequestError::NonTextContent),
        };
        let tool_calls = message
            .tool_calls
            .into_iter()
            .map(|call| {
                let arguments: serde_json::Value = serde_json::from_str(&call.function.arguments)
                    .map_err(|_| {
                    RequestError::InvalidToolArguments(call.function.name.clone())
                })?;
                if !arguments.is_object() {
                    return Err(RequestError::InvalidToolArguments(call.function.name));
                }
                Ok(ToolCall {
                    id: call.id,
                    kind: call.kind,
                    function: FunctionCall {
                        name: call.function.name,
                        arguments,
                    },
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ChatMessage {
            role: match message.role {
                OpenAiRole::System | OpenAiRole::Developer => ChatRole::System,
                OpenAiRole::User => ChatRole::User,
                OpenAiRole::Assistant => ChatRole::Assistant,
                OpenAiRole::Tool => ChatRole::Tool,
            },
            content,
            tool_calls,
            tool_call_id: message.tool_call_id,
        })
    }
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<Model>,
}

#[derive(Serialize)]
pub struct Model {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: OpenAiError,
}

#[derive(Serialize)]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub code: Option<&'static str>,
}

impl ErrorResponse {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                kind: "invalid_request_error",
                code: None,
            },
        }
    }

    pub fn server_error(message: impl Into<String>) -> Self {
        Self {
            error: OpenAiError {
                message: message.into(),
                kind: "server_error",
                code: None,
            },
        }
    }
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: FinishReason,
}

#[derive(Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OpenAiToolCallResponse>,
}

#[derive(Serialize)]
pub struct OpenAiToolCallResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: OpenAiFunctionCallResponse,
}

#[derive(Serialize)]
pub struct OpenAiFunctionCallResponse {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl ChatCompletionResponse {
    pub fn from_generate(model: String, response: GenerateResponse) -> Self {
        let created = created();
        let tool_calls = tool_calls(response.tool_calls);
        Self {
            id: format!("chatcmpl-{}", response.request_id),
            object: "chat.completion",
            created,
            model,
            choices: vec![CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: (!response.text.is_empty()).then_some(response.text),
                    tool_calls,
                },
                finish_reason: response.finish_reason,
            }],
            usage: usage(response.input_tokens, response.output_tokens),
        }
    }
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CompletionUsage>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: AssistantDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Serialize, Default)]
pub struct AssistantDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChunkToolCall>,
}

#[derive(Serialize)]
pub struct ChunkToolCall {
    pub index: usize,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: OpenAiFunctionCallResponse,
}

pub struct StreamContext {
    id: String,
    created: u64,
    model: String,
}

impl StreamContext {
    pub fn new(model: String, request_id: Uuid) -> Self {
        Self {
            id: format!("chatcmpl-{request_id}"),
            created: created(),
            model,
        }
    }

    pub fn role(&self) -> ChatCompletionChunk {
        self.chunk(
            AssistantDelta {
                role: Some("assistant"),
                ..Default::default()
            },
            None,
            None,
        )
    }

    pub fn text(&self, text: String) -> ChatCompletionChunk {
        self.chunk(
            AssistantDelta {
                content: Some(text),
                ..Default::default()
            },
            None,
            None,
        )
    }

    pub fn completed(
        &self,
        input_tokens: usize,
        output_tokens: usize,
        calls: Vec<ToolCall>,
        finish_reason: FinishReason,
    ) -> Vec<ChatCompletionChunk> {
        let mut chunks = Vec::new();
        if !calls.is_empty() {
            chunks.push(
                self.chunk(
                    AssistantDelta {
                        tool_calls: tool_calls(calls)
                            .into_iter()
                            .enumerate()
                            .map(|(index, call)| ChunkToolCall {
                                index,
                                id: call.id,
                                kind: call.kind,
                                function: call.function,
                            })
                            .collect(),
                        ..Default::default()
                    },
                    None,
                    None,
                ),
            );
        }
        chunks.push(self.chunk(
            AssistantDelta::default(),
            Some(finish_reason),
            Some(usage(input_tokens, output_tokens)),
        ));
        chunks
    }

    fn chunk(
        &self,
        delta: AssistantDelta,
        finish_reason: Option<FinishReason>,
        usage: Option<CompletionUsage>,
    ) -> ChatCompletionChunk {
        chunk(
            &self.id,
            self.created,
            &self.model,
            delta,
            finish_reason,
            usage,
        )
    }
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: AssistantDelta,
    finish_reason: Option<FinishReason>,
    usage: Option<CompletionUsage>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.into(),
        object: "chat.completion.chunk",
        created,
        model: model.into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
    }
}

fn tool_calls(calls: Vec<ToolCall>) -> Vec<OpenAiToolCallResponse> {
    calls
        .into_iter()
        .map(|call| OpenAiToolCallResponse {
            id: call.id,
            kind: call.kind,
            function: OpenAiFunctionCallResponse {
                name: call.function.name,
                arguments: call.function.arguments.to_string(),
            },
        })
        .collect()
}

fn usage(input_tokens: usize, output_tokens: usize) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens: input_tokens + output_tokens,
    }
}

fn created() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_openai_tool_history() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "tiny-model",
            "messages": [
                {"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_time","arguments":"{\"timezone\":\"UTC\"}"}}]},
                {"role":"tool","tool_call_id":"call-1","content":"12:00"}
            ],
            "tools": [{"type":"function","function":{"name":"get_time","parameters":{"type":"object"}}}]
        })).unwrap();
        let request = GenerateRequest::try_from(request).unwrap();
        assert_eq!(
            request.messages.unwrap()[0].tool_calls[0]
                .function
                .arguments["timezone"],
            "UTC"
        );
    }

    #[test]
    fn serializes_tool_completion() {
        let response = ChatCompletionResponse::from_generate(
            "tiny-model".into(),
            GenerateResponse {
                request_id: Uuid::nil(),
                worker_id: "worker-a".into(),
                model: "tiny-model".into(),
                text: String::new(),
                input_tokens: 3,
                output_tokens: 4,
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    kind: ToolType::Function,
                    function: FunctionCall {
                        name: "get_time".into(),
                        arguments: serde_json::json!({"timezone":"UTC"}),
                    },
                }],
                finish_reason: FinishReason::ToolCalls,
            },
        );
        assert_eq!(
            response.choices[0].message.tool_calls[0].function.arguments,
            r#"{"timezone":"UTC"}"#
        );
    }
}
