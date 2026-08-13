use anyhow::Result;
use decompute_sdk::{ChatRequest, GenerationConfig, GenerationEvent, GgufModelHandle};
use protocol::{
    Acceleration, GenerateRequest, GenerateResponse, GenerationStreamEvent, ModelCapability,
    ModelManifest, ModelStatus, WorkerCapabilities, WorkerState,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::mpsc;

pub struct WorkerRuntime {
    pub node_id: String,
    pub model_id: String,
    pub address: String,
    pub max_requests: usize,
    pub manifest: ModelManifest,
    active: AtomicUsize,
    draining: AtomicBool,
    model: GgufModelHandle,
    acceleration: Acceleration,
}

impl WorkerRuntime {
    pub fn new(
        node_id: String,
        model_id: String,
        address: String,
        max_requests: usize,
        manifest: ModelManifest,
        acceleration: Acceleration,
        model: GgufModelHandle,
    ) -> Self {
        Self {
            node_id,
            model_id,
            address,
            max_requests,
            manifest,
            active: AtomicUsize::new(0),
            draining: AtomicBool::new(false),
            model,
            acceleration,
        }
    }

    pub fn capabilities(&self) -> WorkerCapabilities {
        let active_requests = self.active.load(Ordering::Acquire);
        let state = if self.draining.load(Ordering::Acquire) {
            WorkerState::Draining
        } else if active_requests >= self.max_requests {
            WorkerState::Busy
        } else {
            WorkerState::Available
        };
        WorkerCapabilities {
            node_id: self.node_id.clone(),
            models: vec![ModelCapability {
                id: self.model_id.clone(),
                status: ModelStatus::Loaded,
                manifest_sha256: Some(self.manifest.id.clone()),
            }],
            active_requests,
            max_requests: self.max_requests,
            state,
            hardware: crate::resources::detect(self.acceleration.clone()),
        }
    }

    pub fn drain(&self) {
        self.draining.store(true, Ordering::Release);
    }
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Acquire)
    }
    pub fn try_reserve(&self, model: &str) -> Result<(), &'static str> {
        if model != self.model_id {
            return Err("requested model is not loaded");
        }
        if self.is_draining() {
            return Err("worker is draining");
        }
        let mut active = self.active.load(Ordering::Acquire);
        loop {
            if active >= self.max_requests {
                return Err("worker is at capacity");
            }
            match self.active.compare_exchange_weak(
                active,
                active + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(value) => active = value,
            }
        }
    }
    pub fn release(&self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        tracing::info!(request_id = %request.request_id, model = %request.model, "starting local inference request");
        let generated = self.model.generate(Self::chat_request(&request)?).await?;
        Ok(GenerateResponse {
            request_id: request.request_id,
            worker_id: self.node_id.clone(),
            model: request.model,
            text: generated.text,
            input_tokens: generated.input_tokens,
            output_tokens: generated.output_tokens,
            tool_calls: generated.tool_calls,
            finish_reason: generated.finish_reason,
        })
    }

    pub async fn stream(
        &self,
        request: GenerateRequest,
        events: mpsc::Sender<GenerationStreamEvent>,
    ) {
        tracing::info!(request_id = %request.request_id, model = %request.model, "starting streamed local inference request");
        let request_id = request.request_id;
        let chat_request = match Self::chat_request(&request) {
            Ok(request) => request,
            Err(error) => {
                let _ = events
                    .send(GenerationStreamEvent::Error {
                        message: format!("{error:#}"),
                    })
                    .await;
                return;
            }
        };
        let mut receiver = match self.model.stream(chat_request).await {
            Ok(receiver) => receiver,
            Err(error) => {
                let _ = events
                    .send(GenerationStreamEvent::Error {
                        message: format!("{error:#}"),
                    })
                    .await;
                return;
            }
        };
        while let Some(event) = receiver.recv().await {
            let event = match event {
                GenerationEvent::TextDelta(text) => GenerationStreamEvent::TextDelta { text },
                GenerationEvent::Completed(result) => {
                    tracing::info!(request_id = %request_id, output_tokens = result.output_tokens, "completed streamed local inference request");
                    GenerationStreamEvent::Completed {
                        input_tokens: result.input_tokens,
                        output_tokens: result.output_tokens,
                        tool_calls: result.tool_calls,
                        finish_reason: result.finish_reason,
                    }
                }
                GenerationEvent::Error(error) => GenerationStreamEvent::Error {
                    message: format!("{error:#}"),
                },
            };
            if events.send(event).await.is_err() {
                break;
            }
        }
    }

    fn chat_request(request: &GenerateRequest) -> Result<ChatRequest> {
        Ok(ChatRequest {
            request_id: request.request_id,
            messages: request.normalized_messages().map_err(anyhow::Error::msg)?,
            template: request.template.clone(),
            tools: request.tools.clone(),
            generation: GenerationConfig {
                max_tokens: request.max_tokens,
                temperature: None,
            },
        })
    }
}
