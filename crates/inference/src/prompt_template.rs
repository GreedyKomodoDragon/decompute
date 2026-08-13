use crate::descriptor::ModelDescriptor;
use anyhow::{Context, Result, bail};
use minijinja::{AutoEscape, Environment, UndefinedBehavior, context};
use protocol::{ChatMessage, ToolDefinition};
use serde::Deserialize;
use std::{collections::BTreeMap, fs};

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<serde_json::Value>,
    eos_token: Option<String>,
}

/// A closed runtime MiniJinja environment containing every template packaged by a model.
pub struct TemplateBundle {
    environment: Environment<'static>,
    names: Vec<String>,
    eos_token: Option<String>,
}

impl TemplateBundle {
    pub fn load(descriptor: &ModelDescriptor) -> Result<Self> {
        let config: TokenizerConfig =
            serde_json::from_reader(fs::File::open(&descriptor.tokenizer_config_path)?)
                .with_context(|| format!("parse {}", descriptor.tokenizer_config_path.display()))?;
        let mut templates = metadata_templates(config.chat_template)?;
        let standalone = descriptor.directory.join("chat_template.jinja");
        if standalone.exists() {
            templates.insert(
                "default".into(),
                fs::read_to_string(&standalone)
                    .with_context(|| format!("read {}", standalone.display()))?,
            );
        }
        if !templates.contains_key("default") {
            bail!(
                "model has no default chat template; available templates: {}",
                template_names(&templates)
            );
        }

        let mut environment = Environment::new();
        environment.set_auto_escape_callback(|_| AutoEscape::None);
        environment.set_undefined_behavior(UndefinedBehavior::SemiStrict);
        for (name, source) in &templates {
            environment
                .add_template_owned(name.clone(), source.clone())
                .with_context(|| format!("compile chat template {name}"))?;
        }
        Ok(Self {
            environment,
            names: templates.into_keys().collect(),
            eos_token: config.eos_token,
        })
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        template: Option<&str>,
        tools: &[ToolDefinition],
    ) -> Result<String> {
        let name = template.unwrap_or("default");
        if !self.names.iter().any(|available| available == name) {
            bail!(
                "unknown chat template {name}; available templates: {}",
                self.names.join(", ")
            );
        }
        self.environment
            .get_template(name)
            .context("load compiled chat template")?
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
                tools => tools,
                eos_token => self.eos_token.as_deref(),
            })
            .with_context(|| format!("render chat template {name}"))
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.eos_token.as_deref()
    }
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

fn metadata_templates(value: Option<serde_json::Value>) -> Result<BTreeMap<String, String>> {
    match value {
        Some(serde_json::Value::String(template)) => {
            Ok(BTreeMap::from([("default".into(), template)]))
        }
        Some(serde_json::Value::Object(templates)) => templates
            .into_iter()
            .map(|(name, value)| match value {
                serde_json::Value::String(source) => Ok((name, source)),
                _ => bail!("chat template {name} must be a string"),
            })
            .collect(),
        Some(_) => bail!("chat_template must be a string or an object of named templates"),
        None => Ok(BTreeMap::new()),
    }
}

fn template_names(templates: &BTreeMap<String, String>) -> String {
    let names = templates
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    if names.is_empty() {
        "none".into()
    } else {
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::ChatRole;
    use std::path::PathBuf;

    fn descriptor(directory: &std::path::Path) -> ModelDescriptor {
        ModelDescriptor {
            directory: directory.into(),
            model_type: "qwen2".into(),
            declared_precision: None,
            stored_precision: crate::execution::ModelPrecision::F32,
            config_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            tokenizer_config_path: directory.join("tokenizer_config.json"),
            weight_paths: vec![],
        }
    }

    fn message() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".into(),
            tool_calls: vec![],
            tool_call_id: None,
        }]
    }

    #[test]
    fn renders_default_template_from_metadata() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"eos_token":"<eos>","chat_template":"{{ messages[0].content }}{{ eos_token }}"}"#,
        )
        .unwrap();
        assert_eq!(
            TemplateBundle::load(&descriptor(directory.path()))
                .unwrap()
                .render(&message(), None, &[])
                .unwrap(),
            "Hello<eos>"
        );
    }

    #[test]
    fn loads_named_templates_and_allows_includes() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": {
                    "default": "{% include 'prefix' %}{{ messages[0].content }}",
                    "prefix": "default:",
                    "rag": "rag:{{ messages[0].content }}"
                }
            })
            .to_string(),
        )
        .unwrap();
        let bundle = TemplateBundle::load(&descriptor(directory.path())).unwrap();
        assert_eq!(
            bundle.render(&message(), None, &[]).unwrap(),
            "default:Hello"
        );
        assert_eq!(
            bundle.render(&message(), Some("rag"), &[]).unwrap(),
            "rag:Hello"
        );
    }

    #[test]
    fn standalone_file_overrides_default() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"chat_template":"metadata"}"#,
        )
        .unwrap();
        fs::write(
            directory.path().join("chat_template.jinja"),
            "file:{{ messages[0].content }}",
        )
        .unwrap();
        assert_eq!(
            TemplateBundle::load(&descriptor(directory.path()))
                .unwrap()
                .render(&message(), None, &[])
                .unwrap(),
            "file:Hello"
        );
    }

    #[test]
    fn passes_tool_definitions_to_templates() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"chat_template":"{% for tool in tools %}{{ tool.function.name }}{% endfor %}"}"#,
        )
        .unwrap();
        let tools = vec![ToolDefinition {
            kind: protocol::ToolType::Function,
            function: protocol::FunctionDefinition {
                name: "get_time".into(),
                description: None,
                parameters: serde_json::json!({"type":"object"}),
            },
        }];
        assert_eq!(
            TemplateBundle::load(&descriptor(directory.path()))
                .unwrap()
                .render(&message(), None, &tools)
                .unwrap(),
            "get_time"
        );
    }
}
