use crate::{
    openai::{
        self, ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, Model, ModelsResponse,
    },
    registry::Registry,
};
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
use futures_util::{StreamExt, stream};
use protocol::{GenerateRequest, GenerateResponse, HeartbeatRequest, RegisterWorkerRequest};
use std::{convert::Infallible, sync::Arc};
use tracing::info;

pub fn router(registry: Arc<Registry>) -> Router {
    Router::new()
        .route("/workers/register", post(register))
        .route("/workers/{id}/heartbeat", post(heartbeat))
        .route("/workers", get(list))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(registry)
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

async fn models(State(registry): State<Arc<Registry>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: registry
            .available_models()
            .await
            .into_iter()
            .map(|id| Model {
                id,
                object: "model",
                owned_by: "decompute",
            })
            .collect(),
    })
}

async fn chat_completions(
    State(registry): State<Arc<Registry>>,
    Json(public): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let stream = public.stream;
    let model = public.model.clone();
    let request = match GenerateRequest::try_from(public) {
        Ok(request) => request,
        Err(err) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                ErrorResponse::invalid_request(err.to_string()),
            )
            .into_response();
        }
    };
    let response = match route_request(&registry, request).await {
        Ok(response) => response,
        Err((status, message)) => {
            return openai_error(status, ErrorResponse::server_error(message)).into_response();
        }
    };
    if stream {
        let events = stream::iter(openai::chunks(model, response))
            .map(|chunk| {
                Ok::<_, Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                )
            })
            .chain(stream::iter([Ok(Event::default().data("[DONE]"))]));
        Sse::new(events).into_response()
    } else {
        Json(ChatCompletionResponse::from_generate(model, response)).into_response()
    }
}

async fn route_request(
    registry: &Arc<Registry>,
    request: GenerateRequest,
) -> Result<GenerateResponse, (StatusCode, String)> {
    let Some(worker) = registry.select_and_reserve(&request.model).await else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "no eligible worker has the requested model".into(),
        ));
    };
    info!(request_id = %request.request_id, worker = %worker.id, model = %request.model, "routing OpenAI-compatible inference request");
    let result = reqwest::Client::new()
        .post(format!("{}/generate", worker.address.trim_end_matches('/')))
        .json(&request)
        .send()
        .await;
    registry.release(&worker.id).await;
    match result {
        Ok(response) if response.status().is_success() => {
            response.json::<GenerateResponse>().await.map_err(|err| {
                (
                    StatusCode::BAD_GATEWAY,
                    format!("invalid worker response: {err}"),
                )
            })
        }
        Ok(response) => Err((
            StatusCode::BAD_GATEWAY,
            format!("worker returned {}", response.status()),
        )),
        Err(err) => {
            registry.mark_offline(&worker.id).await;
            Err((
                StatusCode::BAD_GATEWAY,
                format!("worker request failed: {err}"),
            ))
        }
    }
}

fn openai_error(status: StatusCode, error: ErrorResponse) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(error))
}
