use crate::state::WorkerRuntime;
use protocol::{HeartbeatRequest, RegisterWorkerRequest};
use std::sync::Arc;
use tokio::time::{Duration, interval};
use tracing::warn;

pub fn start(worker: Arc<WorkerRuntime>, coordinator: String) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let register_url = format!("{}/workers/register", coordinator.trim_end_matches('/'));
        let register = || RegisterWorkerRequest {
            address: worker.address.clone(),
            capabilities: worker.capabilities(),
        };
        if let Err(err) = client.post(&register_url).json(&register()).send().await {
            warn!(%err, "worker registration failed; heartbeats will retry");
        }
        let mut ticker = interval(Duration::from_secs(5));
        loop {
            ticker.tick().await;
            let url = format!(
                "{}/workers/{}/heartbeat",
                coordinator.trim_end_matches('/'),
                worker.node_id
            );
            match client
                .post(url)
                .json(&HeartbeatRequest {
                    capabilities: worker.capabilities(),
                })
                .send()
                .await
            {
                Ok(response) if response.status() == reqwest::StatusCode::NOT_FOUND => {
                    warn!("coordinator forgot this worker; registering again");
                    if let Err(err) = client.post(&register_url).json(&register()).send().await {
                        warn!(%err, "worker re-registration failed");
                    }
                }
                Ok(response) if !response.status().is_success() => {
                    warn!(status = %response.status(), "heartbeat rejected")
                }
                Ok(_) => {}
                Err(err) => warn!(%err, "heartbeat failed"),
            }
        }
    });
}
