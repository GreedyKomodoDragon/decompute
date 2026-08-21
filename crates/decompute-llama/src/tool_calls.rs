use anyhow::{Result, bail};
use decompute_core::{FunctionCall, ToolCall, ToolDefinition, ToolType};
use serde_json::Value;
use std::collections::HashSet;
use uuid::Uuid;

pub fn parse_tool_calls(
    text: &str,
    tools: &[ToolDefinition],
    request_id: Uuid,
) -> Result<(String, Vec<ToolCall>)> {
    let names = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<HashSet<_>>();
    let mut visible = String::new();
    let mut remainder = text;
    let mut calls = Vec::new();
    let mut ids = HashSet::new();
    while let Some(start) = remainder.find("<tool_call>") {
        visible.push_str(&remainder[..start]);
        let body = &remainder[start + "<tool_call>".len()..];
        let Some(end) = body.find("</tool_call>") else {
            bail!("unterminated Qwen <tool_call> block");
        };
        let call = parse_one(body[..end].trim(), &names, request_id, calls.len())?;
        if !ids.insert(call.id.clone()) {
            bail!("model returned duplicate tool-call id {:?}", call.id);
        }
        calls.push(call);
        remainder = &body[end + "</tool_call>".len()..];
    }
    visible.push_str(remainder);
    Ok((visible.trim().to_owned(), calls))
}

fn parse_one(
    body: &str,
    names: &HashSet<&str>,
    request_id: Uuid,
    index: usize,
) -> Result<ToolCall> {
    let value: Value = serde_json::from_str(body)
        .map_err(|error| anyhow::anyhow!("invalid Qwen tool-call JSON: {error}"))?;
    let function = value.get("function").unwrap_or(&value);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("tool call has no function name"))?;
    if !names.contains(name) {
        bail!("model requested undeclared tool {name:?}");
    }
    let arguments = function
        .get("arguments")
        .ok_or_else(|| anyhow::anyhow!("tool call {name:?} has no arguments"))?;
    let arguments = match arguments {
        Value::String(value) => serde_json::from_str(value).map_err(|error| {
            anyhow::anyhow!("tool call {name:?} has invalid JSON arguments: {error}")
        })?,
        value => value.clone(),
    };
    if !arguments.is_object() {
        bail!("tool call {name:?} arguments must be a JSON object");
    }
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{request_id}-{index}"));
    if id.is_empty() {
        bail!("tool call id must not be empty");
    }
    Ok(ToolCall {
        id,
        kind: ToolType::Function,
        function: FunctionCall {
            name: name.to_owned(),
            arguments,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use decompute_core::{FunctionDefinition, ToolType};

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
    fn extracts_qwen_tool_call_and_visible_text() {
        let (text, calls) = parse_tool_calls("<tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}\n</tool_call>", &tools(), Uuid::nil()).unwrap();
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_time");
        assert_eq!(calls[0].function.arguments["timezone"], "UTC");
    }

    #[test]
    fn extracts_multiple_calls_and_preserves_visible_text() {
        let output = concat!(
            "before\n",
            "<tool_call>{\"name\":\"get_time\",\"arguments\":{}}</tool_call>\n",
            "between\n",
            "<tool_call>{\"name\":\"get_time\",\"arguments\":{\"tz\":\"UTC\"}}</tool_call>\n",
            "after"
        );
        let (text, calls) = parse_tool_calls(output, &tools(), Uuid::nil()).unwrap();
        assert_eq!(text, "before\n\nbetween\n\nafter");
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn accepts_string_encoded_arguments() {
        let output =
            r#"<tool_call>{"name":"get_time","arguments":"{\"timezone\":\"UTC\"}"}</tool_call>"#;
        let (_, calls) = parse_tool_calls(output, &tools(), Uuid::nil()).unwrap();
        assert_eq!(calls[0].function.arguments["timezone"], "UTC");
    }

    #[test]
    fn rejects_duplicate_explicit_ids() {
        let output = concat!(
            r#"<tool_call>{"id":"same","name":"get_time","arguments":{}}</tool_call>"#,
            r#"<tool_call>{"id":"same","name":"get_time","arguments":{}}</tool_call>"#
        );
        assert!(parse_tool_calls(output, &tools(), Uuid::nil()).is_err());
    }
}
