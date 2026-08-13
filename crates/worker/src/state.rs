use anyhow::Result;
use inference::{GenerationConfig, LocalModel};
use protocol::{
    Acceleration, GenerateRequest, GenerateResponse, GenerationStreamEvent, ModelCapability,
    ModelManifest, ModelStatus, WorkerCapabilities, WorkerState,
};
use std::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread,
};
use tokio::sync::{mpsc, oneshot};

pub enum InferenceJob {
    Generate {
        request: GenerateRequest,
        response: oneshot::Sender<Result<GenerateResponse>>,
    },
    Stream {
        request: GenerateRequest,
        events: mpsc::Sender<GenerationStreamEvent>,
        finished: oneshot::Sender<()>,
    },
}

pub struct WorkerRuntime {
    pub node_id: String,
    pub model_id: String,
    pub address: String,
    pub max_requests: usize,
    pub manifest: ModelManifest,
    active: AtomicUsize,
    draining: AtomicBool,
    jobs: mpsc::Sender<InferenceJob>,
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
        mut model: LocalModel,
    ) -> Self {
        let (jobs, mut receiver) = mpsc::channel(max_requests.max(1));
        let runtime = Self {
            node_id,
            model_id,
            address,
            max_requests,
            manifest,
            active: AtomicUsize::new(0),
            draining: AtomicBool::new(false),
            jobs,
            acceleration,
        };
        thread::Builder::new()
            .name("local-inference".into())
            .spawn(move || {
                while let Some(job) = receiver.blocking_recv() {
                    match job {
                        InferenceJob::Generate { request, response } => {
                            tracing::info!(request_id = %request.request_id, model = %request.model, "inference thread started request");
                            let result = request
                                .normalized_messages()
                                .map_err(anyhow::Error::msg)
                                .and_then(|messages| {
                                    model.generate(
                                        &messages,
                                        request.template.as_deref(),
                                        &request.tools,
                                        request.request_id,
                                        GenerationConfig {
                                            max_tokens: request.max_tokens,
                                            temperature: None,
                                        },
                                    )
                                })
                                .map(|generated| GenerateResponse {
                                    request_id: request.request_id,
                                    worker_id: String::new(),
                                    model: request.model,
                                    text: generated.text,
                                    input_tokens: generated.input_tokens,
                                    output_tokens: generated.output_tokens,
                                    tool_calls: generated.tool_calls,
                                    finish_reason: generated.finish_reason,
                                });
                            let _ = response.send(result);
                        }
                        InferenceJob::Stream {
                            request,
                            events,
                            finished,
                        } => {
                            tracing::info!(request_id = %request.request_id, model = %request.model, "inference thread started streamed request");
                            let messages = match request.normalized_messages() {
                                Ok(messages) => messages,
                                Err(err) => {
                                    tracing::error!(error = %format!("{err:#}"), "streamed inference failed");
                                    let _ = events.blocking_send(GenerationStreamEvent::Error {
                                        message: format!("{err:#}"),
                                    });
                                    let _ = finished.send(());
                                    continue;
                                }
                            };
                            let mut callback = |token: &str| {
                                events
                                    .blocking_send(GenerationStreamEvent::TextDelta {
                                        text: token.to_owned(),
                                    })
                                    .map_err(|_| anyhow::anyhow!("stream client disconnected"))
                            };
                            match model.generate_with_callback(
                                &messages,
                                request.template.as_deref(),
                                &request.tools,
                                request.request_id,
                                GenerationConfig {
                                    max_tokens: request.max_tokens,
                                    temperature: None,
                                },
                                &mut callback,
                            ) {
                                Ok(generated) => {
                                    tracing::info!(
                                        request_id = %request.request_id,
                                        output_tokens = generated.output_tokens,
                                        tool_calls = generated.tool_calls.len(),
                                        finish_reason = ?generated.finish_reason,
                                        "inference thread completed streamed request"
                                    );
                                    let _ =
                                        events.blocking_send(GenerationStreamEvent::Completed {
                                            input_tokens: generated.input_tokens,
                                            output_tokens: generated.output_tokens,
                                            tool_calls: generated.tool_calls,
                                            finish_reason: generated.finish_reason,
                                        });
                                }
                                Err(err) => {
                                    tracing::error!(
                                        request_id = %request.request_id,
                                        error = %format!("{err:#}"),
                                        "inference thread failed streamed request"
                                    );
                                    let _ = events.blocking_send(GenerationStreamEvent::Error {
                                        message: format!("{err:#}"),
                                    });
                                }
                            }
                            let _ = finished.send(());
                        }
                    }
                }
            })
            .expect("start inference thread");
        runtime
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
    pub async fn submit(&self, job: InferenceJob) -> Result<(), &'static str> {
        self.jobs
            .send(job)
            .await
            .map_err(|_| "inference worker stopped")
    }
}
