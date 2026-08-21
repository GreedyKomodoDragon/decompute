use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionDefinition,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ToolHistoryError {
    #[error("tool calls are only valid on assistant messages")]
    ToolCallsOnNonAssistant,
    #[error("tool call ids must be non-empty")]
    EmptyToolCallId,
    #[error("tool call ids must be unique within a response")]
    DuplicateToolCallId,
    #[error("assistant messages may not include tool_call_id")]
    AssistantHasToolCallId,
    #[error("tool call name {0:?} was not declared")]
    UndeclaredTool(String),
    #[error("tool results must include a tool_call_id")]
    MissingToolCallId,
    #[error("tool results may not include tool_calls")]
    ToolResultHasToolCalls,
    #[error("tool call arguments must be a JSON object")]
    InvalidToolArguments,
    #[error("tool result {actual:?} must follow tool call {expected:?}")]
    ToolResultOutOfOrder { expected: String, actual: String },
    #[error("tool result references unknown or already-consumed call {0:?}")]
    UnknownToolCallId(String),
    #[error("tool call history has unresolved calls")]
    UnresolvedToolCalls,
}

/// Validates the assistant/tool message protocol before a message history is
/// rendered into a model prompt. Tool implementations remain outside the
/// worker; this only checks that client-supplied results match model requests.
pub fn validate_tool_history(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
) -> Result<(), ToolHistoryError> {
    let declared = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<HashSet<_>>();
    let mut pending = VecDeque::<String>::new();
    let mut pending_ids = HashSet::<String>::new();

    for message in messages {
        match message.role {
            ChatRole::Assistant => {
                if message.tool_call_id.is_some() {
                    return Err(ToolHistoryError::AssistantHasToolCallId);
                }
                let mut response_ids = HashSet::new();
                for call in &message.tool_calls {
                    if call.id.is_empty() {
                        return Err(ToolHistoryError::EmptyToolCallId);
                    }
                    if !response_ids.insert(&call.id) || pending_ids.contains(&call.id) {
                        return Err(ToolHistoryError::DuplicateToolCallId);
                    }
                    if !declared.contains(call.function.name.as_str()) {
                        return Err(ToolHistoryError::UndeclaredTool(call.function.name.clone()));
                    }
                    if !call.function.arguments.is_object() {
                        return Err(ToolHistoryError::InvalidToolArguments);
                    }
                    pending.push_back(call.id.clone());
                    pending_ids.insert(call.id.clone());
                }
            }
            ChatRole::Tool => {
                if !message.tool_calls.is_empty() {
                    return Err(ToolHistoryError::ToolResultHasToolCalls);
                }
                let id = message
                    .tool_call_id
                    .as_deref()
                    .filter(|id| !id.is_empty())
                    .ok_or(ToolHistoryError::MissingToolCallId)?;
                let Some(expected) = pending.front() else {
                    return Err(ToolHistoryError::UnknownToolCallId(id.to_owned()));
                };
                if expected != id {
                    if pending_ids.contains(id) {
                        return Err(ToolHistoryError::ToolResultOutOfOrder {
                            expected: expected.clone(),
                            actual: id.to_owned(),
                        });
                    }
                    return Err(ToolHistoryError::UnknownToolCallId(id.to_owned()));
                }
                pending.pop_front();
                pending_ids.remove(id);
            }
            ChatRole::System | ChatRole::User => {
                if !message.tool_calls.is_empty() || message.tool_call_id.is_some() {
                    return Err(ToolHistoryError::ToolCallsOnNonAssistant);
                }
                if !pending.is_empty() {
                    return Err(ToolHistoryError::UnresolvedToolCalls);
                }
            }
        }
    }

    if pending.is_empty() {
        Ok(())
    } else {
        Err(ToolHistoryError::UnresolvedToolCalls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> ToolDefinition {
        ToolDefinition {
            kind: ToolType::Function,
            function: FunctionDefinition {
                name: "get_time".into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
            },
        }
    }

    fn call() -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            kind: ToolType::Function,
            function: FunctionCall {
                name: "get_time".into(),
                arguments: serde_json::json!({"timezone": "UTC"}),
            },
        }
    }

    #[test]
    fn accepts_complete_tool_round_trip() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "What time is it?".into(),
                tool_calls: vec![],
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: String::new(),
                tool_calls: vec![call()],
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::Tool,
                content: "12:00 UTC".into(),
                tool_calls: vec![],
                tool_call_id: Some("call-1".into()),
            },
        ];
        assert!(validate_tool_history(&messages, &[tool()]).is_ok());
    }

    #[test]
    fn rejects_unmatched_tool_result() {
        let message = ChatMessage {
            role: ChatRole::Tool,
            content: "12:00 UTC".into(),
            tool_calls: vec![],
            tool_call_id: Some("missing".into()),
        };
        assert_eq!(
            validate_tool_history(&[message], &[tool()]),
            Err(ToolHistoryError::UnknownToolCallId("missing".into()))
        );
    }

    #[test]
    fn rejects_tool_results_out_of_order() {
        let mut second = call();
        second.id = "call-2".into();
        let messages = vec![
            ChatMessage {
                role: ChatRole::Assistant,
                content: String::new(),
                tool_calls: vec![call(), second],
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::Tool,
                content: "second".into(),
                tool_calls: vec![],
                tool_call_id: Some("call-2".into()),
            },
        ];
        assert_eq!(
            validate_tool_history(&messages, &[tool()]),
            Err(ToolHistoryError::ToolResultOutOfOrder {
                expected: "call-1".into(),
                actual: "call-2".into(),
            })
        );
    }

    #[test]
    fn rejects_unresolved_tool_call() {
        let message = ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            tool_calls: vec![call()],
            tool_call_id: None,
        };
        assert_eq!(
            validate_tool_history(&[message], &[tool()]),
            Err(ToolHistoryError::UnresolvedToolCalls)
        );
    }
}
