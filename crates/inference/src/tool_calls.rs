use anyhow::{Context, Result, bail};
use protocol::{FinishReason, FunctionCall, ToolCall, ToolDefinition, ToolType};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct ParsedToolCalls {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
}

pub trait ToolCallParser: Send + Sync {
    fn parse(
        &self,
        generated: &str,
        tools: &[ToolDefinition],
        request_id: Uuid,
    ) -> Result<ParsedToolCalls>;
}

pub struct PlainTextToolCallParser;

impl ToolCallParser for PlainTextToolCallParser {
    fn parse(
        &self,
        generated: &str,
        _tools: &[ToolDefinition],
        _request_id: Uuid,
    ) -> Result<ParsedToolCalls> {
        Ok(ParsedToolCalls {
            text: generated.to_owned(),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
        })
    }
}

pub struct QwenToolCallParser;

#[derive(Deserialize)]
struct RawToolCall {
    name: String,
    arguments: serde_json::Value,
}

impl ToolCallParser for QwenToolCallParser {
    fn parse(
        &self,
        generated: &str,
        tools: &[ToolDefinition],
        request_id: Uuid,
    ) -> Result<ParsedToolCalls> {
        const START: &str = "<tool_call>";
        const END: &str = "</tool_call>";
        let mut remaining = generated;
        let mut text = String::new();
        let mut calls = Vec::new();
        loop {
            let Some(start) = remaining.find(START) else {
                if remaining.contains(END) {
                    bail!("Qwen tool call output contains closing tag without opening tag");
                }
                text.push_str(remaining);
                break;
            };
            text.push_str(&remaining[..start]);
            let payload_start = start + START.len();
            let after_start = &remaining[payload_start..];
            let Some(end) = after_start.find(END) else {
                bail!("Qwen tool call output is missing closing tag");
            };
            let raw: RawToolCall = serde_json::from_str(after_start[..end].trim())
                .context("parse Qwen tool call JSON")?;
            if !raw.arguments.is_object() {
                bail!(
                    "Qwen tool call arguments for {} must be a JSON object",
                    raw.name
                );
            }
            if !tools.iter().any(|tool| tool.function.name == raw.name) {
                bail!("Qwen requested unknown tool {}", raw.name);
            }
            calls.push(ToolCall {
                id: format!("{request_id}-{}", calls.len()),
                kind: ToolType::Function,
                function: FunctionCall {
                    name: raw.name,
                    arguments: raw.arguments,
                },
            });
            remaining = &after_start[end + END.len()..];
        }
        let finish_reason = if calls.is_empty() {
            FinishReason::Stop
        } else {
            FinishReason::ToolCalls
        };
        Ok(ParsedToolCalls {
            text,
            tool_calls: calls,
            finish_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::FunctionDefinition;

    fn tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            kind: ToolType::Function,
            function: FunctionDefinition {
                name: "get_time".into(),
                description: None,
                parameters: serde_json::json!({"type":"object"}),
            },
        }]
    }

    #[test]
    fn parses_qwen_tool_call() {
        let parsed = QwenToolCallParser.parse("<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}\n</tool_call>", &tools(), Uuid::nil()).unwrap();
        assert_eq!(parsed.finish_reason, FinishReason::ToolCalls);
        assert_eq!(parsed.tool_calls[0].function.name, "get_time");
    }

    #[test]
    fn rejects_unknown_tool() {
        assert!(
            QwenToolCallParser
                .parse(
                    "<tool_call>{\"name\":\"shell\",\"arguments\":{}}</tool_call>",
                    &tools(),
                    Uuid::nil()
                )
                .is_err()
        );
    }
}
