use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

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
    pub max_tokens: usize,
}

impl GenerateRequest {
    pub fn normalized_messages(&self) -> Result<Vec<ChatMessage>, RequestValidationError> {
        normalize_messages(self.prompt.as_deref(), self.messages.as_deref())
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicGenerateRequest {
    pub request_id: Option<Uuid>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ChatMessage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

impl PublicGenerateRequest {
    pub fn into_generate_request(self) -> Result<GenerateRequest, RequestValidationError> {
        let messages = normalize_messages(self.prompt.as_deref(), self.messages.as_deref())?;
        Ok(GenerateRequest {
            request_id: self.request_id.unwrap_or_else(Uuid::new_v4),
            model: self.model,
            prompt: None,
            messages: Some(messages),
            template: self.template,
            max_tokens: self.max_tokens,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RequestValidationError {
    #[error("provide exactly one of prompt or messages")]
    ExactlyOneOfPromptOrMessages,
    #[error("messages must not be empty")]
    EmptyMessages,
}

fn normalize_messages(
    prompt: Option<&str>,
    messages: Option<&[ChatMessage]>,
) -> Result<Vec<ChatMessage>, RequestValidationError> {
    match (prompt, messages) {
        (Some(prompt), None) => Ok(vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.to_owned(),
        }]),
        (None, Some(messages)) if !messages.is_empty() => Ok(messages.to_vec()),
        (None, Some(_)) => Err(RequestValidationError::EmptyMessages),
        _ => Err(RequestValidationError::ExactlyOneOfPromptOrMessages),
    }
}

const fn default_max_tokens() -> usize {
    100
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicGenerateResponse {
    pub request_id: Uuid,
    pub worker_id: String,
    pub model: String,
    pub text: String,
    pub usage: Usage,
}

impl From<GenerateResponse> for PublicGenerateResponse {
    fn from(response: GenerateResponse) -> Self {
        Self {
            request_id: response.request_id,
            worker_id: response.worker_id,
            model: response.model,
            text: response.text,
            usage: Usage {
                input_tokens: response.input_tokens,
                output_tokens: response.output_tokens,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenEvent {
    pub token: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_normalizes_to_one_user_message() {
        assert_eq!(
            normalize_messages(Some("hello"), None).unwrap(),
            vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".into()
            }]
        );
    }

    #[test]
    fn requires_exactly_one_input_shape() {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
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
    fn preserves_selected_template_when_normalizing() {
        let request = PublicGenerateRequest {
            request_id: None,
            model: "tiny-model".into(),
            prompt: Some("hello".into()),
            messages: None,
            template: Some("rag".into()),
            max_tokens: 10,
        };
        assert_eq!(
            request.into_generate_request().unwrap().template.as_deref(),
            Some("rag")
        );
    }
}
