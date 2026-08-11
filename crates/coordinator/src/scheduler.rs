use crate::registry::WorkerRecord;
use protocol::{ModelStatus, WorkerState};

pub fn select_worker<'a>(
    workers: impl Iterator<Item = &'a WorkerRecord>,
    requested_model: &str,
) -> Option<&'a WorkerRecord> {
    workers
        .filter(|worker| worker.state == WorkerState::Available)
        .filter(|worker| {
            worker
                .models
                .iter()
                .any(|model| model.id == requested_model && model.status == ModelStatus::Loaded)
        })
        .filter(|worker| worker.active_requests < worker.max_requests)
        .min_by_key(|worker| worker.active_requests)
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{Acceleration, HardwareInfo, ModelCapability};
    use std::time::Instant;
    fn worker(id: &str, active: usize, state: WorkerState, model: &str) -> WorkerRecord {
        WorkerRecord {
            id: id.into(),
            address: "http://x".into(),
            models: vec![ModelCapability {
                id: model.into(),
                status: ModelStatus::Loaded,
                manifest_sha256: None,
            }],
            active_requests: active,
            max_requests: 2,
            last_heartbeat: Instant::now(),
            state,
            hardware: HardwareInfo {
                architecture: "arm64".into(),
                total_memory_bytes: 0,
                available_memory_bytes: 0,
                acceleration: Acceleration::Cpu,
            },
        }
    }
    #[test]
    fn chooses_least_active_eligible_worker() {
        let workers = vec![
            worker("a", 1, WorkerState::Available, "tiny-model"),
            worker("b", 0, WorkerState::Available, "tiny-model"),
            worker("c", 0, WorkerState::Draining, "tiny-model"),
        ];
        assert_eq!(select_worker(workers.iter(), "tiny-model").unwrap().id, "b");
    }
    #[test]
    fn requires_exact_loaded_model() {
        let workers = vec![worker("a", 0, WorkerState::Available, "other")];
        assert!(select_worker(workers.iter(), "tiny-model").is_none());
    }
}
