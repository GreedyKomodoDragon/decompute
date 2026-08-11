use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("worker is unavailable")]
    WorkerUnavailable,
    #[error("requested model is not loaded")]
    ModelUnavailable,
    #[error("worker is draining")]
    Draining,
}

#[derive(Clone, Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
