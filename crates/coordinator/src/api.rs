use crate::registry::Registry;
use async_stream::stream;
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use futures_util::StreamExt;
use protocol::{
    ErrorResponse, GenerateRequest, HeartbeatRequest, PublicGenerateRequest,
    PublicGenerateResponse, RegisterWorkerRequest,
};
use std::{io, sync::Arc};
use tracing::info;
use uuid::Uuid;

pub fn router(registry: Arc<Registry>) -> Router {
    Router::new()
        .route("/workers/register", post(register))
        .route("/workers/{id}/heartbeat", post(heartbeat))
        .route("/workers", get(list))
        .route("/v1/generate", post(generate))
        .route("/v1/generate/stream", post(stream_generate))
        .with_state(registry)
}
fn error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
}
async fn register(
    State(registry): State<Arc<Registry>>,
    Json(request): Json<RegisterWorkerRequest>,
) -> StatusCode {
    registry.register(request).await;
    StatusCode::CREATED
}
async fn heartbeat(
    State(registry): State<Arc<Registry>>,
    Path(id): Path<String>,
    Json(request): Json<HeartbeatRequest>,
) -> StatusCode {
    if registry.heartbeat(&id, request).await {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}
async fn list(State(registry): State<Arc<Registry>>) -> Json<Vec<crate::registry::WorkerRecord>> {
    Json(registry.list().await)
}
fn request(request: PublicGenerateRequest) -> GenerateRequest {
    GenerateRequest {
        request_id: request.request_id.unwrap_or_else(Uuid::new_v4),
        model: request.model,
        prompt: request.prompt,
        max_tokens: request.max_tokens,
    }
}
async fn generate(
    State(registry): State<Arc<Registry>>,
    Json(public): Json<PublicGenerateRequest>,
) -> impl IntoResponse {
    let request = request(public);
    let Some(worker) = registry.select_and_reserve(&request.model).await else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no eligible worker has the requested model",
        )
        .into_response();
    };
    info!(request_id = %request.request_id, worker = %worker.id, model = %request.model, "routing inference request");
    let response = reqwest::Client::new()
        .post(format!("{}/generate", worker.address.trim_end_matches('/')))
        .json(&request)
        .send()
        .await;
    registry.release(&worker.id).await;
    match response {
        Ok(response) if response.status().is_success() => {
            match response.json::<protocol::GenerateResponse>().await {
                Ok(value) => Json(PublicGenerateResponse::from(value)).into_response(),
                Err(err) => error(
                    StatusCode::BAD_GATEWAY,
                    format!("invalid worker response: {err}"),
                )
                .into_response(),
            }
        }
        Ok(response) => error(
            StatusCode::BAD_GATEWAY,
            format!("worker returned {}", response.status()),
        )
        .into_response(),
        Err(err) => {
            registry.mark_offline(&worker.id).await;
            error(
                StatusCode::BAD_GATEWAY,
                format!("worker request failed: {err}"),
            )
            .into_response()
        }
    }
}
async fn stream_generate(
    State(registry): State<Arc<Registry>>,
    Json(public): Json<PublicGenerateRequest>,
) -> impl IntoResponse {
    let request = request(public);
    let Some(worker) = registry.select_and_reserve(&request.model).await else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no eligible worker has the requested model",
        )
        .into_response();
    };
    let response = match reqwest::Client::new()
        .post(format!(
            "{}/generate/stream",
            worker.address.trim_end_matches('/')
        ))
        .json(&request)
        .send()
        .await
    {
        Ok(value) if value.status().is_success() => value,
        Ok(value) => {
            registry.release(&worker.id).await;
            return error(
                StatusCode::BAD_GATEWAY,
                format!("worker returned {}", value.status()),
            )
            .into_response();
        }
        Err(err) => {
            registry.mark_offline(&worker.id).await;
            registry.release(&worker.id).await;
            return error(
                StatusCode::BAD_GATEWAY,
                format!("worker request failed: {err}"),
            )
            .into_response();
        }
    };
    let worker_id = worker.id.clone();
    let release_registry = registry.clone();
    let body = Body::from_stream(
        stream! { let mut bytes = response.bytes_stream(); while let Some(chunk) = bytes.next().await { match chunk { Ok(chunk) => yield Ok::<_, io::Error>(chunk), Err(err) => { yield Err(io::Error::other(err)); break; } } } release_registry.release(&worker_id).await; },
    );
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}
