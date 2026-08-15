use crate::registry::WorkerRecord;
use protocol::{Acceleration, ModelStatus, WorkerState};

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
        .min_by_key(|worker| {
            (
                acceleration_rank(&worker.hardware.acceleration),
                worker.active_requests,
                worker.id.as_str(),
            )
        })
}

fn acceleration_rank(acceleration: &Acceleration) -> u8 {
    match acceleration {
        Acceleration::Metal | Acceleration::Cuda => 0,
        Acceleration::Cpu => 1,
    }
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
    fn prefers_accelerated_worker_over_idle_cpu_worker() {
        let mut metal = worker("metal", 1, WorkerState::Available, "tiny-model");
        metal.hardware.acceleration = Acceleration::Metal;
        let cpu = worker("cpu", 0, WorkerState::Available, "tiny-model");
        assert_eq!(
            select_worker([&cpu, &metal].into_iter(), "tiny-model")
                .unwrap()
                .id,
            "metal"
        );
    }
    #[test]
    fn falls_back_to_cpu_when_accelerated_worker_is_busy() {
        let mut metal = worker("metal", 2, WorkerState::Busy, "tiny-model");
        metal.hardware.acceleration = Acceleration::Metal;
        let cpu = worker("cpu", 0, WorkerState::Available, "tiny-model");
        assert_eq!(
            select_worker([&metal, &cpu].into_iter(), "tiny-model")
                .unwrap()
                .id,
            "cpu"
        );
    }
    #[test]
    fn breaks_equal_ties_by_worker_id() {
        let b = worker("b", 0, WorkerState::Available, "tiny-model");
        let a = worker("a", 0, WorkerState::Available, "tiny-model");
        assert_eq!(
            select_worker([&b, &a].into_iter(), "tiny-model")
                .unwrap()
                .id,
            "a"
        );
    }
    #[test]
    fn requires_exact_loaded_model() {
        let workers = vec![worker("a", 0, WorkerState::Available, "other")];
        assert!(select_worker(workers.iter(), "tiny-model").is_none());
    }
}
