use anyhow::{Context, Result};
use decompute_core::FinishReason;
use decompute_llama::{GgufGenerationConfig, GgufModel};
use std::{path::Path, thread};
use tokio::sync::{mpsc, oneshot};

use crate::{ChatRequest, GenerationEvent, GenerationResult};

pub use decompute_llama::GgufLoadConfig;

/// A cloneable async handle to a GGUF/llama.cpp model.
///
/// Enable this crate's `llama` feature. Enable `llama-metal` to compile the
/// llama.cpp Metal backend on Apple Silicon.
#[derive(Clone)]
pub struct GgufModelHandle {
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

impl GgufModelHandle {
    pub fn load(path: impl AsRef<Path>, config: GgufLoadConfig) -> Result<Self> {
        let mut model = GgufModel::load(path, config).context("load local GGUF model")?;
        let (jobs, mut receiver) = mpsc::channel(1);
        thread::Builder::new()
            .name("decompute-sdk-gguf".into())
            .spawn(move || {
                while let Some(job) = receiver.blocking_recv() {
                    match job {
                        Job::Generate { request, response } => {
                            let result = generate(&mut model, request, &mut |_| Ok(()));
                            let _ = response.send(result);
                        }
                        Job::Stream { request, events } => {
                            let callback_events = events.clone();
                            let mut callback = move |text: &str| {
                                callback_events
                                    .blocking_send(GenerationEvent::TextDelta(text.to_owned()))
                                    .map_err(|_| anyhow::anyhow!("stream receiver dropped"))
                            };
                            match generate(&mut model, request, &mut callback) {
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
            .context("start GGUF model execution thread")?;
        Ok(Self { jobs })
    }

    pub async fn generate(&self, request: ChatRequest) -> Result<GenerationResult> {
        let (response_tx, response_rx) = oneshot::channel();
        self.jobs
            .send(Job::Generate {
                request,
                response: response_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("GGUF model execution thread stopped"))?;
        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("GGUF model execution response dropped"))?
    }

    pub async fn stream(&self, request: ChatRequest) -> Result<mpsc::Receiver<GenerationEvent>> {
        let (events, receiver) = mpsc::channel(32);
        self.jobs
            .send(Job::Stream { request, events })
            .await
            .map_err(|_| anyhow::anyhow!("GGUF model execution thread stopped"))?;
        Ok(receiver)
    }
}

fn generate(
    model: &mut GgufModel,
    request: ChatRequest,
    callback: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<GenerationResult> {
    let result = model.generate_chat(
        &request.messages,
        GgufGenerationConfig {
            max_tokens: request.generation.max_tokens,
            temperature: request.generation.temperature.map(|value| value as f32),
        },
        callback,
    )?;
    Ok(GenerationResult {
        text: result.text,
        input_tokens: result.input_tokens,
        output_tokens: result.output_tokens,
        tool_calls: vec![],
        finish_reason: FinishReason::Stop,
    })
}
