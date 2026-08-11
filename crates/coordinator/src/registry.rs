use protocol::{HeartbeatRequest, RegisterWorkerRequest, WorkerState};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;

#[derive(Clone, Debug, serde::Serialize)]
pub struct WorkerRecord {
    pub id: String,
    pub address: String,
    pub models: Vec<protocol::ModelCapability>,
    pub active_requests: usize,
    pub max_requests: usize,
    #[serde(skip_serializing)]
    pub last_heartbeat: Instant,
    pub state: WorkerState,
    pub hardware: protocol::HardwareInfo,
}

#[derive(Clone, Debug)]
pub struct SelectedWorker {
    pub id: String,
    pub address: String,
}

#[derive(Default)]
pub struct Registry {
    workers: RwLock<HashMap<String, WorkerRecord>>,
}

impl Registry {
    pub async fn register(&self, request: RegisterWorkerRequest) {
        let c = request.capabilities;
        self.workers.write().await.insert(
            c.node_id.clone(),
            WorkerRecord {
                id: c.node_id,
                address: request.address,
                models: c.models,
                active_requests: c.active_requests,
                max_requests: c.max_requests,
                last_heartbeat: Instant::now(),
                state: c.state,
                hardware: c.hardware,
            },
        );
    }
    pub async fn heartbeat(&self, id: &str, request: HeartbeatRequest) -> bool {
        let c = request.capabilities;
        let mut workers = self.workers.write().await;
        let Some(record) = workers.get_mut(id) else {
            return false;
        };
        record.models = c.models;
        record.max_requests = c.max_requests;
        record.active_requests = c.active_requests;
        record.state = c.state;
        record.hardware = c.hardware;
        record.last_heartbeat = Instant::now();
        true
    }
    pub async fn list(&self) -> Vec<WorkerRecord> {
        self.workers.read().await.values().cloned().collect()
    }
    pub async fn select_and_reserve(&self, model: &str) -> Option<SelectedWorker> {
        let mut workers = self.workers.write().await;
        let id = crate::scheduler::select_worker(workers.values(), model)?
            .id
            .clone();
        let worker = workers.get_mut(&id)?;
        worker.active_requests += 1;
        if worker.active_requests >= worker.max_requests {
            worker.state = WorkerState::Busy;
        }
        Some(SelectedWorker {
            id: worker.id.clone(),
            address: worker.address.clone(),
        })
    }
    pub async fn release(&self, id: &str) {
        if let Some(worker) = self.workers.write().await.get_mut(id) {
            worker.active_requests = worker.active_requests.saturating_sub(1);
            if worker.state == WorkerState::Busy && worker.active_requests < worker.max_requests {
                worker.state = WorkerState::Available;
            }
        }
    }
    pub async fn mark_offline(&self, id: &str) {
        if let Some(worker) = self.workers.write().await.get_mut(id) {
            worker.state = WorkerState::Offline;
        }
    }
    pub async fn expire_stale(&self, timeout: Duration) {
        self.expire_at(Instant::now(), timeout).await;
    }
    async fn expire_at(&self, now: Instant, timeout: Duration) {
        for worker in self.workers.write().await.values_mut() {
            if now.duration_since(worker.last_heartbeat) > timeout {
                worker.state = WorkerState::Offline;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{
        Acceleration, HardwareInfo, ModelCapability, ModelStatus, RegisterWorkerRequest,
        WorkerCapabilities,
    };
    fn registration() -> RegisterWorkerRequest {
        RegisterWorkerRequest {
            address: "http://worker".into(),
            capabilities: WorkerCapabilities {
                node_id: "a".into(),
                models: vec![ModelCapability {
                    id: "tiny-model".into(),
                    status: ModelStatus::Loaded,
                    manifest_sha256: None,
                }],
                active_requests: 0,
                max_requests: 1,
                state: WorkerState::Available,
                hardware: HardwareInfo {
                    architecture: "arm64".into(),
                    total_memory_bytes: 1,
                    available_memory_bytes: 1,
                    acceleration: Acceleration::Cpu,
                },
            },
        }
    }
    #[tokio::test]
    async fn expiration_marks_worker_offline() {
        let registry = Registry::default();
        registry.register(registration()).await;
        registry
            .expire_at(
                Instant::now() + Duration::from_secs(16),
                Duration::from_secs(15),
            )
            .await;
        assert_eq!(registry.list().await[0].state, WorkerState::Offline);
    }
}
