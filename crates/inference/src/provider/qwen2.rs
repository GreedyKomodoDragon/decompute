use super::{CausalModelBackend, ModelProvider};
use crate::descriptor::ModelDescriptor;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2;
use std::fs;

pub struct Qwen2Provider;

impl ModelProvider for Qwen2Provider {
    fn model_types(&self) -> &'static [&'static str] {
        &["qwen2"]
    }

    fn load(
        &self,
        descriptor: &ModelDescriptor,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn CausalModelBackend>> {
        let config: qwen2::Config =
            serde_json::from_reader(fs::File::open(&descriptor.config_path)?)
                .context("parse Qwen2 config.json")?;
        let weights = descriptor
            .weight_paths
            .first()
            .context("Qwen2 requires a safetensors weight file")?;
        let builder = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device) }
            .context("memory-map Qwen2 safetensors")?;
        Ok(Box::new(Qwen2Backend {
            model: qwen2::ModelForCausalLM::new(&config, builder).context("load Qwen2 model")?,
        }))
    }
}

struct Qwen2Backend {
    model: qwen2::ModelForCausalLM,
}

impl CausalModelBackend for Qwen2Backend {
    fn next_token_logits(&mut self, input_ids: &Tensor, sequence_offset: usize) -> Result<Tensor> {
        self.model
            .forward(input_ids, sequence_offset)
            .context("Qwen2 forward pass")?
            .squeeze(0)
            .context("remove Qwen2 batch dimension")?
            .squeeze(0)
            .context("remove Qwen2 sequence dimension")
    }
    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
