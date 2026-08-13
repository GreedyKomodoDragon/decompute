use anyhow::{Context, Result, bail};
use decompute_core::{FinishReason, FunctionCall, ToolCall, ToolDefinition, ToolType};
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

    fn stream_filter(&self) -> Box<dyn StreamingOutputFilter>;
}

/// Removes provider control markup before it crosses the worker HTTP boundary.
/// The complete raw output is still parsed at generation completion.
pub trait StreamingOutputFilter: Send {
    fn push(&mut self, fragment: &str) -> String;
    fn finish(&mut self) -> String;
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

    fn stream_filter(&self) -> Box<dyn StreamingOutputFilter> {
        Box::<PlainTextStreamFilter>::default()
    }
}

#[derive(Default)]
struct PlainTextStreamFilter;

impl StreamingOutputFilter for PlainTextStreamFilter {
    fn push(&mut self, fragment: &str) -> String {
        fragment.to_owned()
    }

    fn finish(&mut self) -> String {
        String::new()
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

    fn stream_filter(&self) -> Box<dyn StreamingOutputFilter> {
        Box::<QwenStreamFilter>::default()
    }
}

#[derive(Default)]
struct QwenStreamFilter {
    pending: String,
    inside_tool_call: bool,
}

impl StreamingOutputFilter for QwenStreamFilter {
    fn push(&mut self, fragment: &str) -> String {
        const START: &str = "<tool_call>";
        const END: &str = "</tool_call>";
        self.pending.push_str(fragment);
        let mut visible = String::new();
        loop {
            let marker = if self.inside_tool_call { END } else { START };
            if let Some(position) = self.pending.find(marker) {
                if !self.inside_tool_call {
                    visible.push_str(&self.pending[..position]);
                }
                self.pending.drain(..position + marker.len());
                self.inside_tool_call = !self.inside_tool_call;
                continue;
            }
            let retained = marker_prefix_suffix_len(&self.pending, marker);
            let emit_len = self.pending.len() - retained;
            if !self.inside_tool_call {
                visible.push_str(&self.pending[..emit_len]);
            }
            self.pending.drain(..emit_len);
            return visible;
        }
    }

    fn finish(&mut self) -> String {
        if self.inside_tool_call {
            self.pending.clear();
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

fn marker_prefix_suffix_len(value: &str, marker: &str) -> usize {
    (1..marker.len().min(value.len()) + 1)
        .rev()
        .find(|length| value.ends_with(&marker[..*length]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use decompute_core::FunctionDefinition;

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

    #[test]
    fn stream_filter_hides_fragmented_qwen_tool_markup() {
        let mut filter = QwenToolCallParser.stream_filter();
        assert_eq!(filter.push("Before <tool"), "Before ");
        assert_eq!(filter.push("_call>{\"name\":\"get_time\"}"), "");
        assert_eq!(filter.push("</tool_call> after"), " after");
        assert_eq!(filter.finish(), "");
    }
}
