use protocol::{HeartbeatRequest, ModelStatus, RegisterWorkerRequest, WorkerState};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use uuid::Uuid;

const AFFINITY_TTL: Duration = Duration::from_secs(15 * 60);

#[derive(Clone, Debug)]
struct AffinityBinding {
    worker_id: String,
    last_used: Instant,
}

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
    affinity: RwLock<HashMap<(Uuid, String), AffinityBinding>>,
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
    pub async fn available_models(&self) -> Vec<String> {
        let mut models = self
            .workers
            .read()
            .await
            .values()
            .filter(|worker| worker.state != WorkerState::Offline)
            .flat_map(|worker| worker.models.iter())
            .filter(|model| model.status == ModelStatus::Loaded)
            .map(|model| model.id.clone())
            .collect::<Vec<_>>();
        models.sort();
        models.dedup();
        models
    }
    pub async fn select_and_reserve(
        &self,
        model: &str,
        session_id: Option<Uuid>,
    ) -> Option<SelectedWorker> {
        let mut workers = self.workers.write().await;
        let now = Instant::now();
        let key = session_id.map(|id| (id, model.to_owned()));
        let preferred = if let Some(key) = key.as_ref() {
            let affinity = self.affinity.read().await;
            affinity
                .get(key)
                .filter(|binding| now.duration_since(binding.last_used) <= AFFINITY_TTL)
                .map(|binding| binding.worker_id.clone())
        } else {
            None
        };
        let id = preferred
            .filter(|id| {
                workers
                    .get(id)
                    .is_some_and(|worker| eligible(worker, model))
            })
            .or_else(|| {
                crate::scheduler::select_worker(workers.values(), model)
                    .map(|worker| worker.id.clone())
            })?;
        if let Some(key) = key {
            self.affinity.write().await.insert(
                key,
                AffinityBinding {
                    worker_id: id.clone(),
                    last_used: now,
                },
            );
        }
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
        self.affinity
            .write()
            .await
            .retain(|_, binding| binding.worker_id != id);
    }
    pub async fn expire_stale(&self, timeout: Duration) {
        self.expire_at(Instant::now(), timeout).await;
    }
    async fn expire_at(&self, now: Instant, timeout: Duration) {
        let mut workers = self.workers.write().await;
        for worker in workers.values_mut() {
            if now.duration_since(worker.last_heartbeat) > timeout {
                worker.state = WorkerState::Offline;
            }
        }
        self.affinity.write().await.retain(|_, binding| {
            now.duration_since(binding.last_used) <= AFFINITY_TTL
                && workers
                    .get(&binding.worker_id)
                    .is_some_and(|worker| worker.state != WorkerState::Offline)
        });
    }
}

fn eligible(worker: &WorkerRecord, model: &str) -> bool {
    worker.state == WorkerState::Available
        && worker.active_requests < worker.max_requests
        && worker
            .models
            .iter()
            .any(|candidate| candidate.id == model && candidate.status == ModelStatus::Loaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{
        Acceleration, HardwareInfo, ModelCapability, ModelStatus, RegisterWorkerRequest,
        WorkerCapabilities,
    };
    fn registration(id: &str, model: &str) -> RegisterWorkerRequest {
        RegisterWorkerRequest {
            address: "http://worker".into(),
            capabilities: WorkerCapabilities {
                node_id: id.into(),
                models: vec![ModelCapability {
                    id: model.into(),
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
        registry.register(registration("a", "tiny-model")).await;
        registry
            .expire_at(
                Instant::now() + Duration::from_secs(16),
                Duration::from_secs(15),
            )
            .await;
        assert_eq!(registry.list().await[0].state, WorkerState::Offline);
    }

    #[tokio::test]
    async fn affinity_prefers_bound_worker_and_rebinds_when_busy() {
        let registry = Registry::default();
        registry.register(registration("a", "tiny-model")).await;
        registry.register(registration("b", "tiny-model")).await;
        let session = Uuid::new_v4();

        let first = registry
            .select_and_reserve("tiny-model", Some(session))
            .await
            .unwrap();
        assert_eq!(first.id, "a");

        let second = registry
            .select_and_reserve("tiny-model", Some(session))
            .await
            .unwrap();
        assert_eq!(second.id, "b");
        assert_eq!(
            registry.affinity.read().await[&(session, "tiny-model".into())].worker_id,
            "b"
        );
    }

    #[tokio::test]
    async fn offline_worker_binding_is_removed_and_falls_back() {
        let registry = Registry::default();
        registry.register(registration("a", "tiny-model")).await;
        registry.register(registration("b", "tiny-model")).await;
        let session = Uuid::new_v4();

        let first = registry
            .select_and_reserve("tiny-model", Some(session))
            .await
            .unwrap();
        registry.release(&first.id).await;
        registry.mark_offline(&first.id).await;

        let fallback = registry
            .select_and_reserve("tiny-model", Some(session))
            .await
            .unwrap();
        assert_eq!(fallback.id, "b");
        assert_eq!(
            registry.affinity.read().await[&(session, "tiny-model".into())].worker_id,
            "b"
        );
    }

    #[tokio::test]
    async fn affinity_requires_the_bound_model() {
        let registry = Registry::default();
        registry.register(registration("a", "other-model")).await;
        registry.register(registration("b", "tiny-model")).await;
        let session = Uuid::new_v4();

        let selected = registry
            .select_and_reserve("tiny-model", Some(session))
            .await
            .unwrap();
        assert_eq!(selected.id, "b");
    }
}
