use crate::state::{InferenceJob, WorkerRuntime};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use protocol::{ErrorResponse, GenerateRequest, TokenEvent};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tracing::info;

pub fn router(state: Arc<WorkerRuntime>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/capabilities", get(capabilities))
        .route("/generate", post(generate))
        .route("/generate/stream", post(stream))
        .route("/drain", post(drain))
        .with_state(state)
}
async fn health() -> &'static str {
    "ok"
}
async fn capabilities(
    State(state): State<Arc<WorkerRuntime>>,
) -> Json<protocol::WorkerCapabilities> {
    Json(state.capabilities())
}
async fn drain(State(state): State<Arc<WorkerRuntime>>) -> Json<protocol::WorkerCapabilities> {
    state.drain();
    Json(state.capabilities())
}
fn error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
}
async fn generate(
    State(state): State<Arc<WorkerRuntime>>,
    Json(request): Json<GenerateRequest>,
) -> impl IntoResponse {
    if let Err(message) = state.try_reserve(&request.model) {
        return error(StatusCode::SERVICE_UNAVAILABLE, message).into_response();
    }
    let (tx, rx) = oneshot::channel();
    info!(request_id = %request.request_id, model = %request.model, "accepted inference request");
    if let Err(message) = state
        .submit(InferenceJob::Generate {
            request,
            response: tx,
        })
        .await
    {
        state.release();
        return error(StatusCode::SERVICE_UNAVAILABLE, message).into_response();
    }
    let result = rx.await;
    state.release();
    match result {
        Ok(Ok(mut response)) => {
            response.worker_id = state.node_id.clone();
            Json(response).into_response()
        }
        Ok(Err(err)) => error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
        Err(_) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference response dropped",
        )
        .into_response(),
    }
}
async fn stream(
    State(state): State<Arc<WorkerRuntime>>,
    Json(request): Json<GenerateRequest>,
) -> impl IntoResponse {
    if let Err(message) = state.try_reserve(&request.model) {
        return error(StatusCode::SERVICE_UNAVAILABLE, message).into_response();
    }
    let (tx, rx) = mpsc::channel::<Result<TokenEvent, String>>(32);
    let (finished_tx, finished_rx) = oneshot::channel();
    if let Err(message) = state
        .submit(InferenceJob::Stream {
            request,
            events: tx,
            finished: finished_tx,
        })
        .await
    {
        state.release();
        return error(StatusCode::SERVICE_UNAVAILABLE, message).into_response();
    }
    let release = state.clone();
    let stream = ReceiverStream::new(rx)
        .map(move |event| {
            let event = match event {
                Ok(token) => {
                    Event::default().data(serde_json::to_string(&token).unwrap_or_default())
                }
                Err(message) => Event::default().event("error").data(message),
            };
            Ok::<_, Infallible>(event)
        })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));
    tokio::spawn(async move {
        let _ = finished_rx.await;
        release.release();
    });
    Sse::new(stream).into_response()
}
