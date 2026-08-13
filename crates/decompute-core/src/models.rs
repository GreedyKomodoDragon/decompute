use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: String,
    pub architecture: String,
    pub revision: String,
    pub quantization: Option<String>,
    pub files: Vec<ModelFile>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Acceleration {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub architecture: String,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub acceleration: Acceleration,
}
