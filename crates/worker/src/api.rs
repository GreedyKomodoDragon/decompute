use crate::state::WorkerRuntime;
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
use protocol::{ErrorResponse, GenerateRequest, GenerationStreamEvent};
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
    Json(mut request): Json<GenerateRequest>,
) -> impl IntoResponse {
    if let Err(err) = normalize_request(&mut request) {
        return error(StatusCode::BAD_REQUEST, err).into_response();
    }
    if let Err(message) = state.try_reserve(&request.model) {
        return error(StatusCode::SERVICE_UNAVAILABLE, message).into_response();
    }
    info!(request_id = %request.request_id, model = %request.model, "accepted inference request");
    let result = state.generate(request).await;
    state.release();
    match result {
        Ok(response) => Json(response).into_response(),
        Err(err) => {
            tracing::error!(error = %format!("{err:#}"), "inference failed");
            error(StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response()
        }
    }
}
async fn stream(
    State(state): State<Arc<WorkerRuntime>>,
    Json(mut request): Json<GenerateRequest>,
) -> impl IntoResponse {
    if let Err(err) = normalize_request(&mut request) {
        return error(StatusCode::BAD_REQUEST, err).into_response();
    }
    if let Err(message) = state.try_reserve(&request.model) {
        return error(StatusCode::SERVICE_UNAVAILABLE, message).into_response();
    }
    let (tx, rx) = mpsc::channel::<GenerationStreamEvent>(32);
    let (finished_tx, finished_rx) = oneshot::channel();
    info!(request_id = %request.request_id, model = %request.model, "accepted streamed inference request");
    let worker = state.clone();
    tokio::spawn(async move {
        worker.stream(request, tx).await;
        let _ = finished_tx.send(());
    });
    let release = state.clone();
    let stream = ReceiverStream::new(rx)
        .map(move |event| {
            Ok::<_, Infallible>(
                Event::default().data(serde_json::to_string(&event).unwrap_or_default()),
            )
        })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));
    tokio::spawn(async move {
        let _ = finished_rx.await;
        release.release();
    });
    Sse::new(stream).into_response()
}

fn normalize_request(request: &mut GenerateRequest) -> Result<(), String> {
    request.validate_tools().map_err(|err| err.to_string())?;
    let messages = request
        .normalized_messages()
        .map_err(|err| err.to_string())?;
    request.prompt = None;
    request.messages = Some(messages);
    Ok(())
}
