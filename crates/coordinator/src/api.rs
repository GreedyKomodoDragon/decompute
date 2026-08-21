use crate::{
    openai::{
        self, ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, Model, ModelsResponse,
    },
    registry::{Registry, SelectedWorker},
};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use futures_util::StreamExt;
use protocol::{
    GenerateRequest, GenerateResponse, GenerationStreamEvent, HeartbeatRequest,
    RegisterWorkerRequest,
};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
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
    headers: HeaderMap,
    Json(public): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let wants_stream = public.stream;
    let model = public.model.clone();
    let session_id = match parse_session_id(&headers) {
        Ok(session_id) => session_id,
        Err(message) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                ErrorResponse::invalid_request(message),
            )
            .into_response();
        }
    };
    let request = match GenerateRequest::try_from(public) {
        Ok(mut request) => {
            request.session_id = session_id;
            request
        }
        Err(err) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                ErrorResponse::invalid_request(err.to_string()),
            )
            .into_response();
        }
    };
    if wants_stream {
        return stream_request(registry, request, model).await;
    }
    let cancellation = CancellationToken::new();
    let guard = cancellation.clone().drop_guard();
    let response = match route_request(&registry, request, cancellation).await {
        Ok(response) => response,
        Err((status, message)) => {
            return openai_error(status, ErrorResponse::server_error(message)).into_response();
        }
    };
    guard.disarm();
    Json(ChatCompletionResponse::from_generate(model, response)).into_response()
}

async fn stream_request(
    registry: Arc<Registry>,
    request: GenerateRequest,
    model: String,
) -> Response {
    let Some(worker) = registry
        .select_and_reserve(&request.model, request.session_id)
        .await
    else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorResponse::server_error("no eligible worker has the requested model"),
        )
        .into_response();
    };
    info!(request_id = %request.request_id, worker = %worker.id, model = %request.model, "starting streamed OpenAI-compatible inference request");
    let response = match reqwest::Client::new()
        .post(format!(
            "{}/generate/stream",
            worker.address.trim_end_matches('/')
        ))
        .json(&request)
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => response,
        Ok(response) => {
            registry.release(&worker.id).await;
            return openai_error(
                StatusCode::BAD_GATEWAY,
                ErrorResponse::server_error(format!("worker returned {}", response.status())),
            )
            .into_response();
        }
        Err(err) => {
            registry.mark_offline(&worker.id).await;
            registry.release(&worker.id).await;
            return openai_error(
                StatusCode::BAD_GATEWAY,
                ErrorResponse::server_error(format!("worker request failed: {err}")),
            )
            .into_response();
        }
    };
    let context = openai::StreamContext::new(model, request.request_id);
    let (events, receiver) = mpsc::channel(32);
    let cancellation = CancellationToken::new();
    info!(request_id = %request.request_id, worker = %worker.id, "worker accepted private stream; opening public SSE response");
    tokio::spawn(forward_worker_stream(
        response,
        events,
        context,
        registry,
        worker,
        request.request_id,
        cancellation,
    ));
    let events = ReceiverStream::new(receiver).map(Ok::<_, Infallible>);
    Sse::new(events).into_response()
}

async fn forward_worker_stream(
    response: reqwest::Response,
    events: mpsc::Sender<Event>,
    context: openai::StreamContext,
    registry: Arc<Registry>,
    worker: SelectedWorker,
    request_id: uuid::Uuid,
    cancellation: CancellationToken,
) {
    let worker_id = &worker.id;
    let mut bytes = response.bytes_stream();
    let mut pending = Vec::new();
    let mut completed = false;
    let mut first_delta = true;
    if !send_event(&events, json_event(context.role())).await {
        cancel_and_release(&registry, &worker, request_id, cancellation).await;
        return;
    }
    while let Some(chunk) = bytes.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(err) => {
                registry.mark_offline(worker_id).await;
                tracing::error!(worker = %worker_id, error = %err, "worker private stream body failed");
                send_error(&events, format!("worker stream failed: {err}")).await;
                break;
            }
        };
        pending.extend_from_slice(&chunk);
        while let Some(end) = sse_frame_end(&pending) {
            let frame = pending.drain(..end).collect::<Vec<_>>();
            let payload = match sse_data(&frame) {
                Ok(Some(payload)) if payload == "[DONE]" => continue,
                Ok(Some(payload)) => payload,
                Ok(None) => continue,
                Err(message) => {
                    send_error(&events, message).await;
                    cancel_and_release(&registry, &worker, request_id, cancellation).await;
                    return;
                }
            };
            let event: GenerationStreamEvent = match serde_json::from_str(&payload) {
                Ok(event) => event,
                Err(err) => {
                    send_error(&events, format!("invalid worker stream event: {err}")).await;
                    cancel_and_release(&registry, &worker, request_id, cancellation).await;
                    return;
                }
            };
            match event {
                GenerationStreamEvent::TextDelta { text } => {
                    if first_delta {
                        info!(worker = %worker_id, "received first text delta from worker");
                        first_delta = false;
                    }
                    if !send_event(&events, json_event(context.text(text))).await {
                        cancel_and_release(&registry, &worker, request_id, cancellation).await;
                        return;
                    }
                }
                GenerationStreamEvent::Completed {
                    input_tokens,
                    output_tokens,
                    tool_calls,
                    finish_reason,
                } => {
                    info!(
                        worker = %worker_id,
                        input_tokens,
                        output_tokens,
                        tool_calls = tool_calls.len(),
                        finish_reason = ?finish_reason,
                        "worker completed private stream"
                    );
                    for chunk in
                        context.completed(input_tokens, output_tokens, tool_calls, finish_reason)
                    {
                        if !send_event(&events, json_event(chunk)).await {
                            registry.release(worker_id).await;
                            return;
                        }
                    }
                    completed = true;
                }
                GenerationStreamEvent::Error { message } => {
                    tracing::error!(worker = %worker_id, error = %message, "worker reported streamed inference failure");
                    send_error(&events, message).await;
                    registry.release(worker_id).await;
                    return;
                }
            }
        }
    }
    if completed {
        info!(worker = %worker_id, "finished forwarding worker stream");
        let _ = events.send(Event::default().data("[DONE]")).await;
        registry.release(worker_id).await;
    } else {
        tracing::error!(worker = %worker_id, "worker private stream ended before completion");
        send_error(&events, "worker stream ended before completion").await;
        cancel_and_release(&registry, &worker, request_id, cancellation).await;
    }
}

async fn send_event(events: &mpsc::Sender<Event>, event: Event) -> bool {
    events.send(event).await.is_ok()
}

async fn send_error(events: &mpsc::Sender<Event>, message: impl Into<String>) {
    let _ = events
        .send(
            Event::default()
                .event("error")
                .data(serde_json::json!({"error": {"message": message.into()}}).to_string()),
        )
        .await;
}

fn json_event<T: serde::Serialize>(value: T) -> Event {
    Event::default().data(serde_json::to_string(&value).unwrap_or_default())
}

fn sse_frame_end(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| position + 2)
        .or_else(|| {
            bytes
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| position + 4)
        })
}

fn sse_data(frame: &[u8]) -> Result<Option<String>, String> {
    let frame = std::str::from_utf8(frame)
        .map_err(|err| format!("invalid UTF-8 worker SSE frame: {err}"))?;
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>();
    if data.is_empty() {
        Ok(None)
    } else {
        Ok(Some(data.join("\n")))
    }
}

async fn route_request(
    registry: &Arc<Registry>,
    request: GenerateRequest,
    cancellation: CancellationToken,
) -> Result<GenerateResponse, (StatusCode, String)> {
    let Some(worker) = registry
        .select_and_reserve(&request.model, request.session_id)
        .await
    else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "no eligible worker has the requested model".into(),
        ));
    };
    info!(request_id = %request.request_id, worker = %worker.id, model = %request.model, "routing OpenAI-compatible inference request");
    let (result_tx, result_rx) = oneshot::channel();
    tokio::spawn(forward_worker_request(
        request,
        worker,
        registry.clone(),
        cancellation,
        result_tx,
    ));
    result_rx.await.unwrap_or_else(|_| {
        Err((
            StatusCode::BAD_GATEWAY,
            "worker request task stopped".into(),
        ))
    })
}

async fn forward_worker_request(
    request: GenerateRequest,
    worker: SelectedWorker,
    registry: Arc<Registry>,
    cancellation: CancellationToken,
    result_tx: oneshot::Sender<Result<GenerateResponse, (StatusCode, String)>>,
) {
    let request_id = request.request_id;
    let request_future = async {
        let response = reqwest::Client::new()
            .post(format!("{}/generate", worker.address.trim_end_matches('/')))
            .json(&request)
            .send()
            .await;
        match response {
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
    };
    let result = tokio::select! {
        _ = cancellation.cancelled() => {
            cancel_worker_request(&registry, &worker, request_id).await;
            registry.release(&worker.id).await;
            return;
        }
        result = request_future => result,
    };
    registry.release(&worker.id).await;
    let _ = result_tx.send(result);
}

async fn cancel_and_release(
    registry: &Registry,
    worker: &SelectedWorker,
    request_id: uuid::Uuid,
    cancellation: CancellationToken,
) {
    cancellation.cancel();
    cancel_worker_request(registry, worker, request_id).await;
    registry.release(&worker.id).await;
}

async fn cancel_worker_request(
    registry: &Registry,
    worker: &SelectedWorker,
    request_id: uuid::Uuid,
) {
    let result = reqwest::Client::new()
        .post(format!(
            "{}/requests/{request_id}/cancel",
            worker.address.trim_end_matches('/')
        ))
        .send()
        .await;
    if let Err(error) = result {
        tracing::warn!(worker = %worker.id, request_id = %request_id, error = %error, "worker cancellation request failed");
        registry.mark_offline(&worker.id).await;
    }
}

fn openai_error(status: StatusCode, error: ErrorResponse) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(error))
}

fn parse_session_id(headers: &HeaderMap) -> Result<Option<uuid::Uuid>, &'static str> {
    let Some(value) = headers.get("x-decompute-session-id") else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .map_err(|_| "X-Decompute-Session-Id must be a UUID")?;
    uuid::Uuid::parse_str(value.trim())
        .map(Some)
        .map_err(|_| "X-Decompute-Session-Id must be a UUID")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, routing::post};
    use protocol::{
        Acceleration, HardwareInfo, ModelCapability, ModelStatus, RegisterWorkerRequest,
        WorkerCapabilities, WorkerState,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use tokio::net::TcpListener;

    #[test]
    fn parses_worker_sse_frames_split_across_chunks() {
        let mut bytes = b"data: {\"type\":\"text_delta\",\"text\":\"hi\"}".to_vec();
        assert!(sse_frame_end(&bytes).is_none());
        bytes.extend_from_slice(b"\n\n");
        let end = sse_frame_end(&bytes).unwrap();
        assert_eq!(
            sse_data(&bytes[..end]).unwrap().as_deref(),
            Some(r#"{"type":"text_delta","text":"hi"}"#)
        );
    }

    #[test]
    fn accepts_only_uuid_session_headers() {
        let mut headers = HeaderMap::new();
        assert_eq!(parse_session_id(&headers).unwrap(), None);
        headers.insert("x-decompute-session-id", "not-a-session".parse().unwrap());
        assert!(parse_session_id(&headers).is_err());
        headers.insert(
            "x-decompute-session-id",
            uuid::Uuid::new_v4().to_string().parse().unwrap(),
        );
        assert!(parse_session_id(&headers).unwrap().is_some());
    }

    #[tokio::test]
    async fn cancellation_reaches_the_worker_and_releases_its_reservation() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = cancelled.clone();
        let worker = Router::new().route(
            "/requests/{id}/cancel",
            post(move || {
                let worker_cancelled = worker_cancelled.clone();
                async move {
                    worker_cancelled.store(true, Ordering::Release);
                    StatusCode::NO_CONTENT
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, worker).await.unwrap();
        });

        let registry = Registry::default();
        registry
            .register(RegisterWorkerRequest {
                address,
                capabilities: WorkerCapabilities {
                    node_id: "worker-a".into(),
                    models: vec![ModelCapability {
                        id: "tiny-model".into(),
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
                        acceleration: Acceleration::Metal,
                    },
                },
            })
            .await;
        let selected = registry
            .select_and_reserve("tiny-model", None)
            .await
            .expect("worker is eligible");
        let cancellation = CancellationToken::new();

        cancel_and_release(
            &registry,
            &selected,
            uuid::Uuid::new_v4(),
            cancellation.clone(),
        )
        .await;

        assert!(cancellation.is_cancelled());
        assert!(cancelled.load(Ordering::Acquire));
        let worker = registry.list().await.pop().expect("registered worker");
        assert_eq!(worker.active_requests, 0);
        assert_eq!(worker.state, WorkerState::Available);
    }
}
