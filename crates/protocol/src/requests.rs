use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub request_id: Uuid,
    pub model: String,
    pub prompt: String,
    pub max_tokens: usize,
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
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
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
