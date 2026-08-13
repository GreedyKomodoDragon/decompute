use crate::descriptor::ModelDescriptor;
use anyhow::{Result, bail};
use candle_core::DType;
use std::fmt;

/// Precision stored by a Hugging Face model or requested by its config metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelPrecision {
    F32,
    F16,
    BF16,
}

impl ModelPrecision {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "float32" | "f32" => Ok(Self::F32),
            "float16" | "half" | "f16" => Ok(Self::F16),
            "bfloat16" | "bf16" => Ok(Self::BF16),
            value => bail!("unsupported model precision {value}"),
        }
    }

    pub const fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }
}

impl fmt::Display for ModelPrecision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    Metal,
    Cuda,
}

impl fmt::Display for ExecutionTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        })
    }
}

/// A selected runtime dtype with the metadata that justified the choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub target: ExecutionTarget,
    pub stored_precision: ModelPrecision,
    pub declared_precision: Option<ModelPrecision>,
    pub runtime_precision: ModelPrecision,
}

impl ExecutionPlan {
    pub const fn dtype(&self) -> DType {
        self.runtime_precision.dtype()
    }

    pub fn rationale(&self) -> String {
        let declared = self
            .declared_precision
            .map(|precision| precision.to_string())
            .unwrap_or_else(|| "unspecified".into());
        format!(
            "model stored as {}, config declares {}, running {} on {}",
            self.stored_precision, declared, self.runtime_precision, self.target
        )
    }
}

pub fn resolve_execution_plan(
    descriptor: &ModelDescriptor,
    target: ExecutionTarget,
    supported_runtime_precisions: &[ModelPrecision],
) -> Result<ExecutionPlan> {
    let stored = descriptor.stored_precision;
    let runtime = supported_runtime_precisions
        .iter()
        .copied()
        .find(|precision| *precision == stored)
        .or_else(|| {
            // Conversion is deliberately explicit and ordered by the provider's
            // declared safe runtime formats rather than by a device-wide guess.
            supported_runtime_precisions.first().copied()
        })
        .ok_or_else(|| anyhow::anyhow!("{} has no supported runtime precisions", target))?;
    Ok(ExecutionPlan {
        target,
        stored_precision: stored,
        declared_precision: descriptor.declared_precision,
        runtime_precision: runtime,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn descriptor(stored_precision: ModelPrecision) -> ModelDescriptor {
        ModelDescriptor {
            directory: PathBuf::new(),
            model_type: "qwen2".into(),
            declared_precision: Some(stored_precision),
            stored_precision,
            config_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            tokenizer_config_path: PathBuf::new(),
            weight_paths: vec![],
        }
    }

    #[test]
    fn cpu_promotes_bfloat_weights_to_its_declared_safe_precision() {
        let plan = resolve_execution_plan(
            &descriptor(ModelPrecision::BF16),
            ExecutionTarget::Cpu,
            &[ModelPrecision::F32],
        )
        .unwrap();
        assert_eq!(plan.runtime_precision, ModelPrecision::F32);
    }

    #[test]
    fn metal_uses_f16_when_bfloat_is_not_a_declared_safe_runtime_precision() {
        let plan = resolve_execution_plan(
            &descriptor(ModelPrecision::BF16),
            ExecutionTarget::Metal,
            &[ModelPrecision::F16, ModelPrecision::F32],
        )
        .unwrap();
        assert_eq!(plan.runtime_precision, ModelPrecision::F16);
    }
}
