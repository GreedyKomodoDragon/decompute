use anyhow::{Context, Result, bail};
use decompute_core::{ChatMessage, ToolDefinition};
use minijinja::{Environment, context};
use std::{fs, path::Path};

const QWEN_TOOLS_TEMPLATE: &str = r#"{%- if tools %}
{{- '<|im_start|>\n' }}
{%- if messages[0].role == 'system' %}{{- messages[0].content }}{%- else %}{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}{%- endif %}
{{- '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }}
{%- for tool in tools %}{{- '\n' }}{{- tool | tojson }}{%- endfor %}
{{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'user' or (message.role == 'system' and not loop.first) or (message.role == 'assistant' and not message.tool_calls) %}
{{- '<|im_start|>' ~ message.role ~ '\n' ~ message.content ~ '<|im_end|>\n' }}
{%- elif message.role == 'assistant' %}
{{- '<|im_start|>assistant' }}{%- if message.content %}{{- '\n' ~ message.content }}{%- endif %}
{%- for call in message.tool_calls %}{{- '\n<tool_call>\n{"name": "' ~ call.function.name ~ '", "arguments": ' ~ (call.function.arguments | tojson) ~ '}\n</tool_call>' }}{%- endfor %}
{{- '<|im_end|>\n' }}
{%- elif message.role == 'tool' %}
{%- if loop.first or messages[loop.index0 - 1].role != 'tool' %}{{- '<|im_start|>user' }}{%- endif %}
{{- '\n<tool_response>\n' ~ message.content ~ '\n</tool_response>' }}
{%- if loop.last or messages[loop.index0 + 1].role != 'tool' %}{{- '<|im_end|>\n' }}{%- endif %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"#;

/// Renders explicit model templates. The GGUF embedded template remains the
/// default; this registry is used for named overrides and Qwen tool calls.
pub struct TemplateRegistry {
    environment: Environment<'static>,
    names: Vec<String>,
    qwen: bool,
}

impl TemplateRegistry {
    pub fn load(model_path: &Path, architecture: &str) -> Result<Self> {
        let mut environment = Environment::new();
        environment.add_template_owned("qwen-tools", QWEN_TOOLS_TEMPLATE)?;
        let mut names = vec!["qwen-tools".into()];
        let directory = sidecar_directory(model_path);
        if directory.is_dir() {
            load_directory(&mut environment, &directory, &directory, &mut names)?;
        }
        names.sort();
        Ok(Self {
            environment,
            names,
            qwen: architecture.to_ascii_lowercase().starts_with("qwen"),
        })
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn select(&self, requested: Option<&str>, has_tools: bool) -> Result<Option<String>> {
        if let Some(name) = requested {
            if self.names.iter().any(|available| available == name) {
                return Ok(Some(name.to_owned()));
            }
            bail!(
                "unknown template {name:?}; available overrides: {}",
                self.names.join(", ")
            );
        }
        if has_tools {
            if self.qwen {
                return Ok(Some("qwen-tools".into()));
            }
            bail!("this GGUF model has tools but no tool-aware template; provide a named override")
        }
        Ok(None)
    }

    pub fn render(
        &self,
        name: &str,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<String> {
        self.environment
            .get_template(name)
            .with_context(|| format!("load template {name:?}"))?
            .render(context!(messages => messages, tools => tools, add_generation_prompt => true))
            .with_context(|| format!("render template {name:?}"))
    }
}

fn sidecar_directory(model_path: &Path) -> std::path::PathBuf {
    let mut path = model_path.as_os_str().to_os_string();
    path.push(".templates");
    path.into()
}

fn load_directory(
    environment: &mut Environment<'static>,
    root: &Path,
    directory: &Path,
    names: &mut Vec<String>,
) -> Result<()> {
    for entry in fs::read_dir(directory).with_context(|| format!("read {}", directory.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            load_directory(environment, root, &path, names)?;
            continue;
        }
        if path.extension().and_then(|value| value.to_str()) != Some("jinja") {
            continue;
        }
        let relative = path.strip_prefix(root).expect("template path under root");
        let name = relative.to_string_lossy().replace('\\', "/");
        let source =
            fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        environment.add_template_owned(name.clone(), source.clone())?;
        if relative.components().count() == 1 {
            let alias = relative
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_owned();
            environment.add_template_owned(alias.clone(), source)?;
            names.push(alias);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use decompute_core::{ChatRole, FunctionDefinition, ToolType};

    #[test]
    fn qwen_tool_template_renders_tool_schema() {
        let registry = TemplateRegistry::load(Path::new("qwen.gguf"), "qwen2").unwrap();
        let output = registry
            .render(
                "qwen-tools",
                &[ChatMessage {
                    role: ChatRole::User,
                    content: "What time is it?".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                }],
                &[ToolDefinition {
                    kind: ToolType::Function,
                    function: FunctionDefinition {
                        name: "get_time".into(),
                        description: None,
                        parameters: serde_json::json!({"type":"object"}),
                    },
                }],
            )
            .unwrap();
        assert!(output.contains("<tools>"));
        assert!(output.contains("get_time"));
        assert!(output.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn named_sidecar_template_can_include_a_subtemplate() {
        let temporary = tempfile::tempdir().unwrap();
        let model = temporary.path().join("model.gguf");
        let templates = temporary.path().join("model.gguf.templates");
        std::fs::create_dir_all(templates.join("partials")).unwrap();
        std::fs::write(
            templates.join("rag.jinja"),
            "{% include 'partials/prefix.jinja' %}{{ messages[0].content }}",
        )
        .unwrap();
        std::fs::write(templates.join("partials/prefix.jinja"), "Context: ").unwrap();
        let registry = TemplateRegistry::load(&model, "llama").unwrap();
        assert_eq!(
            registry.select(Some("rag"), false).unwrap().as_deref(),
            Some("rag")
        );
        let output = registry
            .render(
                "rag",
                &[ChatMessage {
                    role: ChatRole::User,
                    content: "hello".into(),
                    tool_calls: vec![],
                    tool_call_id: None,
                }],
                &[],
            )
            .unwrap();
        assert_eq!(output, "Context: hello");
    }
}
