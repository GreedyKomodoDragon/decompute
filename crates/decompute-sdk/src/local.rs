use anyhow::{Context, Result};
use decompute_core::{ChatMessage, ToolDefinition};
use inference::LocalModel;
use std::{path::Path, thread};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::{GenerationConfig, GenerationResult};

/// A transport-neutral chat-generation request.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub template: Option<String>,
    pub tools: Vec<ToolDefinition>,
    pub request_id: Uuid,
    pub generation: GenerationConfig,
}

/// A progressive result from [`LocalModelHandle::stream`].
#[derive(Debug)]
pub enum GenerationEvent {
    TextDelta(String),
    Completed(GenerationResult),
    Error(anyhow::Error),
}

/// A cloneable async handle to one locally loaded model.
///
/// The model itself remains on a dedicated OS thread, so synchronous Candle
/// execution cannot block an application's async executor. Each model handle
/// processes one request at a time; create more handles to add concurrency.
#[derive(Clone)]
pub struct LocalModelHandle {
    jobs: mpsc::Sender<Job>,
}

enum Job {
    Generate {
        request: ChatRequest,
        response: oneshot::Sender<Result<GenerationResult>>,
    },
    Stream {
        request: ChatRequest,
        events: mpsc::Sender<GenerationEvent>,
    },
}

impl LocalModelHandle {
    /// Loads a safetensors/Candle model and starts its dedicated execution thread.
    pub fn load_candle(path: impl AsRef<Path>) -> Result<Self> {
        let mut model = LocalModel::load(path).context("load local Candle model")?;
        let (jobs, mut receiver) = mpsc::channel(1);
        thread::Builder::new()
            .name("decompute-sdk-inference".into())
            .spawn(move || {
                while let Some(job) = receiver.blocking_recv() {
                    match job {
                        Job::Generate { request, response } => {
                            let result = model
                                .generate(
                                    &request.messages,
                                    request.template.as_deref(),
                                    &request.tools,
                                    request.request_id,
                                    request.generation.into(),
                                )
                                .map(GenerationResult::from);
                            let _ = response.send(result);
                        }
                        Job::Stream { request, events } => {
                            let callback_events = events.clone();
                            let mut callback = move |text: &str| {
                                callback_events
                                    .blocking_send(GenerationEvent::TextDelta(text.to_owned()))
                                    .map_err(|_| anyhow::anyhow!("stream receiver dropped"))
                            };
                            match model
                                .generate_with_callback(
                                    &request.messages,
                                    request.template.as_deref(),
                                    &request.tools,
                                    request.request_id,
                                    request.generation.into(),
                                    &mut callback,
                                )
                                .map(GenerationResult::from)
                            {
                                Ok(result) => {
                                    let _ =
                                        events.blocking_send(GenerationEvent::Completed(result));
                                }
                                Err(error) => {
                                    let _ = events.blocking_send(GenerationEvent::Error(error));
                                }
                            }
                        }
                    }
                }
            })
            .context("start local model execution thread")?;
        Ok(Self { jobs })
    }

    /// Runs a request without blocking the caller's async executor.
    pub async fn generate(&self, request: ChatRequest) -> Result<GenerationResult> {
        let (response_tx, response_rx) = oneshot::channel();
        self.jobs
            .send(Job::Generate {
                request,
                response: response_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("local model execution thread stopped"))?;
        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("local model execution response dropped"))?
    }

    /// Starts a request and returns progressive text plus its terminal result.
    pub async fn stream(&self, request: ChatRequest) -> Result<mpsc::Receiver<GenerationEvent>> {
        let (events, receiver) = mpsc::channel(32);
        self.jobs
            .send(Job::Stream { request, events })
            .await
            .map_err(|_| anyhow::anyhow!("local model execution thread stopped"))?;
        Ok(receiver)
    }
}
