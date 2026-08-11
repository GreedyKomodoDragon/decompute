#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: Option<f64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

pub type TokenCallback<'a> = dyn FnMut(&str) -> anyhow::Result<()> + Send + 'a;
