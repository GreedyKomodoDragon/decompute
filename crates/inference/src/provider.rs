mod qwen2;

use crate::descriptor::ModelDescriptor;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub trait CausalModelBackend: Send {
    /// Returns a rank-1 vocabulary logits tensor for the next generated token.
    fn next_token_logits(&mut self, input_ids: &Tensor, sequence_offset: usize) -> Result<Tensor>;
    fn clear_kv_cache(&mut self);
}

pub trait ModelProvider: Sync {
    fn model_types(&self) -> &'static [&'static str];
    fn load(
        &self,
        descriptor: &ModelDescriptor,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn CausalModelBackend>>;
}

pub struct ProviderRegistry {
    providers: Vec<&'static dyn ModelProvider>,
}

impl ProviderRegistry {
    pub fn builtin() -> Self {
        Self {
            providers: vec![&qwen2::Qwen2Provider],
        }
    }

    pub fn load(
        &self,
        descriptor: &ModelDescriptor,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn CausalModelBackend>> {
        let provider = self
            .providers
            .iter()
            .find(|provider| {
                provider
                    .model_types()
                    .contains(&descriptor.model_type.as_str())
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unsupported model_type {}; supported types: {}",
                    descriptor.model_type,
                    self.supported_model_types().join(", ")
                )
            })?;
        provider.load(descriptor, device, dtype)
    }

    pub fn supported_model_types(&self) -> Vec<&'static str> {
        self.providers
            .iter()
            .flat_map(|provider| provider.model_types().iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn reports_unsupported_model_types() {
        let descriptor = ModelDescriptor {
            directory: PathBuf::new(),
            model_type: "unknown".into(),
            config_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            tokenizer_config_path: PathBuf::new(),
            weight_paths: vec![],
        };
        let error = match ProviderRegistry::builtin().load(&descriptor, &Device::Cpu, DType::F32) {
            Ok(_) => panic!("unknown provider unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unsupported model_type unknown"));
    }
}
