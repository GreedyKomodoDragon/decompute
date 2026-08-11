use crate::{HardwareInfo, ModelCapability};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    Available,
    Busy,
    Draining,
    Offline,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    pub node_id: String,
    pub models: Vec<ModelCapability>,
    pub active_requests: usize,
    pub max_requests: usize,
    pub state: WorkerState,
    pub hardware: HardwareInfo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterWorkerRequest {
    pub address: String,
    #[serde(flatten)]
    pub capabilities: WorkerCapabilities,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    #[serde(flatten)]
    pub capabilities: WorkerCapabilities,
}
