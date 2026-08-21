use crate::state::WorkerRuntime;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use protocol::{ErrorResponse, GenerateRequest, GenerationStreamEvent};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tracing::info;
use uuid::Uuid;

pub fn router(state: Arc<WorkerRuntime>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/capabilities", get(capabilities))
        .route("/generate", post(generate))
        .route("/generate/stream", post(stream))
        .route("/requests/{id}/cancel", post(cancel))
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
async fn cancel(
    State(state): State<Arc<WorkerRuntime>>,
    Path(request_id): Path<Uuid>,
) -> StatusCode {
    if state.cancel_request(request_id).await {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
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
    let request_id = request.request_id;
    let Some(cancellation) = state.begin_request(request_id) else {
        state.release_unstarted_request();
        return error(StatusCode::CONFLICT, "request is already active").into_response();
    };
    let result = state.generate(request, cancellation).await;
    state.finish_request(request_id);
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
    info!(request_id = %request.request_id, model = %request.model, "accepted streamed inference request");
    let request_id = request.request_id;
    let Some(cancellation) = state.begin_request(request_id) else {
        state.release_unstarted_request();
        return error(StatusCode::CONFLICT, "request is already active").into_response();
    };
    let worker = state.clone();
    tokio::spawn(async move {
        worker.stream(request, tx, cancellation).await;
        worker.finish_request(request_id);
    });
    let stream = ReceiverStream::new(rx)
        .map(move |event| {
            Ok::<_, Infallible>(
                Event::default().data(serde_json::to_string(&event).unwrap_or_default()),
            )
        })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));
    Sse::new(stream).into_response()
}

fn normalize_request(request: &mut GenerateRequest) -> Result<(), String> {
    request.validate_tools().map_err(|err| err.to_string())?;
    let messages = request
        .normalized_messages()
        .map_err(|err| err.to_string())?;
    request.prompt = None;
    request.messages = Some(messages);
    request
        .validate_tool_history_messages(request.messages.as_deref().unwrap_or_default())
        .map_err(|err| err.to_string())?;
    Ok(())
}
