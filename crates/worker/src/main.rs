mod api;
mod heartbeat;
mod resources;
mod state;

use anyhow::Result;
use axum::Router;
use clap::Parser;
use decompute_models::{ModelCatalog, ModelSource, resolve};
use decompute_sdk::{GgufLoadConfig, GgufModelHandle};
use protocol::Acceleration;
use state::WorkerRuntime;
use std::{
    fmt, net::SocketAddr, num::NonZeroU32, path::Path, path::PathBuf, str::FromStr, sync::Arc,
};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 9001)]
    port: u16,
    #[arg(long)]
    node_id: String,
    #[arg(long)]
    coordinator: String,
    /// Curated model ID from Decompute's embedded model catalog.
    #[arg(long, default_value = "qwen2.5-0.5b-instruct-q4-k-m")]
    model: String,
    /// Use an already-downloaded copy of the selected catalog model instead of
    /// resolving it through Hugging Face. The file is still checksum-verified.
    #[arg(long)]
    model_path: Option<PathBuf>,
    /// `auto` uses the GGUF model's trained context length; a positive integer
    /// explicitly overrides it.
    #[arg(long, default_value_t = ContextTokens::Auto)]
    context_tokens: ContextTokens,
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    #[arg(long)]
    advertise_address: Option<String>,
    #[arg(long, default_value_t = 1)]
    max_requests: usize,
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    device: DeviceChoice,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum DeviceChoice {
    Auto,
    Cpu,
    Metal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContextTokens {
    Auto,
    Explicit(NonZeroU32),
}

#[derive(Debug)]
struct ContextTokensParseError;

impl fmt::Display for ContextTokensParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("expected `auto` or a positive integer")
    }
}

impl std::error::Error for ContextTokensParseError {}

impl fmt::Display for ContextTokens {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => formatter.write_str("auto"),
            Self::Explicit(tokens) => tokens.fmt(formatter),
        }
    }
}

impl FromStr for ContextTokens {
    type Err = ContextTokensParseError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        value
            .parse::<NonZeroU32>()
            .map(Self::Explicit)
            .map_err(|_| ContextTokensParseError)
    }
}

fn resolve_context_tokens(selection: ContextTokens, trained: NonZeroU32) -> NonZeroU32 {
    match selection {
        ContextTokens::Auto => trained,
        ContextTokens::Explicit(tokens) => tokens,
    }
}

async fn load_model(
    path: &Path,
    device: DeviceChoice,
    context_tokens: NonZeroU32,
) -> Result<(GgufModelHandle, Acceleration)> {
    match device {
        DeviceChoice::Cpu => {
            load_and_verify(
                path,
                load_config(context_tokens, Some(0)),
                Acceleration::Cpu,
            )
            .await
        }
        DeviceChoice::Metal => load_metal(path, context_tokens).await,
        DeviceChoice::Auto => {
            #[cfg(feature = "metal")]
            {
                match load_and_verify(
                    path,
                    load_config(context_tokens, Some(u32::MAX)),
                    Acceleration::Metal,
                )
                .await
                {
                    Ok(model) => return Ok(model),
                    Err(err) => {
                        tracing::warn!(error = %format!("{err:#}"), "Metal model probe failed; falling back to CPU")
                    }
                }
            }
            load_and_verify(
                path,
                load_config(context_tokens, Some(0)),
                Acceleration::Cpu,
            )
            .await
        }
    }
}

fn load_config(context_tokens: NonZeroU32, gpu_layers: Option<u32>) -> GgufLoadConfig {
    GgufLoadConfig {
        context_tokens: context_tokens.get(),
        gpu_layers,
    }
}

async fn load_and_verify(
    path: &Path,
    config: GgufLoadConfig,
    acceleration: Acceleration,
) -> Result<(GgufModelHandle, Acceleration)> {
    let model = GgufModelHandle::load(path, config)?;
    info!(
        target = ?acceleration,
        "loaded GGUF llama.cpp execution plan"
    );
    model.smoke_test().await?;
    Ok((model, acceleration))
}

#[cfg(feature = "metal")]
async fn load_metal(
    path: &Path,
    context_tokens: NonZeroU32,
) -> Result<(GgufModelHandle, Acceleration)> {
    load_and_verify(
        path,
        load_config(context_tokens, Some(u32::MAX)),
        Acceleration::Metal,
    )
    .await
}
#[cfg(not(feature = "metal"))]
async fn load_metal(
    _path: &Path,
    _context_tokens: NonZeroU32,
) -> Result<(GgufModelHandle, Acceleration)> {
    anyhow::bail!("Metal support was not compiled; run with --features metal")
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();
    let address = args
        .advertise_address
        .clone()
        .unwrap_or_else(|| format!("http://{}:{}", args.bind, args.port));
    let catalog = ModelCatalog::embedded()?;
    let entry = catalog.get(&args.model)?.clone();
    info!(model = %entry.id, repository = %entry.repository, revision = %entry.revision, "resolving curated model");
    if args.model_path.is_none() {
        info!(model = %entry.id, "checking Hugging Face cache or downloading model");
    }
    let resolved = resolve(entry, args.model_path.as_deref()).await?;
    info!(
        model = %resolved.entry.id,
        path = %resolved.path.display(),
        source = ?resolved.source,
        "verified curated model"
    );
    if resolved.source == ModelSource::LocalOverride {
        info!(model = %resolved.entry.id, "using verified local model override");
    }
    let model_info = decompute_llama::inspect(&resolved.path)?;
    let context_tokens =
        resolve_context_tokens(args.context_tokens, model_info.trained_context_tokens);
    info!(
        model = %resolved.entry.id,
        requested_context_tokens = %args.context_tokens,
        context_tokens = context_tokens.get(),
        trained_context_tokens = model_info.trained_context_tokens.get(),
        context_source = if args.context_tokens == ContextTokens::Auto { "gguf_metadata" } else { "cli_override" },
        "selected model context capacity"
    );
    info!(model = %resolved.entry.id, "loading local model");
    let (model, acceleration) = load_model(&resolved.path, args.device, context_tokens).await?;
    let runtime = Arc::new(WorkerRuntime::new(
        args.node_id,
        resolved.entry.id,
        address,
        args.max_requests,
        resolved.manifest,
        acceleration,
        model,
    ));
    heartbeat::start(runtime.clone(), args.coordinator);
    let app: Router = api::router(runtime).layer(TraceLayer::new_for_http());
    let bind: SocketAddr = format!("{}:{}", args.bind, args.port).parse()?;
    info!(%bind, "worker listening");
    axum::serve(TcpListener::bind(bind).await?, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_args(arguments: &[&str]) -> Args {
        Args::try_parse_from(arguments).expect("valid worker arguments")
    }

    #[test]
    fn context_tokens_defaults_to_auto_and_uses_model_metadata() {
        let args = parse_args(&[
            "worker",
            "--node-id",
            "worker-a",
            "--coordinator",
            "http://127.0.0.1:8000",
        ]);
        let trained = NonZeroU32::new(32_768).unwrap();
        assert_eq!(args.context_tokens, ContextTokens::Auto);
        assert_eq!(
            resolve_context_tokens(args.context_tokens, trained),
            trained
        );
        assert_eq!(
            load_config(
                resolve_context_tokens(args.context_tokens, trained),
                Some(0)
            )
            .context_tokens,
            trained.get()
        );
        assert_eq!(
            load_config(
                resolve_context_tokens(args.context_tokens, trained),
                Some(u32::MAX)
            )
            .context_tokens,
            trained.get()
        );
    }

    #[test]
    fn context_tokens_accept_explicit_non_zero_values_and_reject_invalid_values() {
        let args = parse_args(&[
            "worker",
            "--node-id",
            "worker-a",
            "--coordinator",
            "http://127.0.0.1:8000",
            "--context-tokens",
            "8192",
        ]);
        assert_eq!(
            args.context_tokens,
            ContextTokens::Explicit(NonZeroU32::new(8_192).unwrap())
        );
        assert_eq!(
            resolve_context_tokens(args.context_tokens, NonZeroU32::new(32_768).unwrap()),
            NonZeroU32::new(8_192).unwrap()
        );
        assert!(
            Args::try_parse_from([
                "worker",
                "--node-id",
                "worker-a",
                "--coordinator",
                "http://127.0.0.1:8000",
                "--context-tokens",
                "0",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from([
                "worker",
                "--node-id",
                "worker-a",
                "--coordinator",
                "http://127.0.0.1:8000",
                "--context-tokens",
                "not-a-context",
            ])
            .is_err()
        );
    }

    #[test]
    fn provider_examples_advertise_the_curated_qwen_context_capacity() {
        let opencode: serde_json::Value =
            serde_json::from_str(include_str!("../../../examples/opencode.json")).unwrap();
        let opencode_models = opencode["provider"]["decompute"]["models"]
            .as_object()
            .unwrap();
        for model in opencode_models.values() {
            assert!(model["limit"]["context"].as_u64().unwrap() == 32_768);
        }

        let pi: serde_json::Value =
            serde_json::from_str(include_str!("../../../examples/pi-models.json")).unwrap();
        let pi_models = pi["providers"]["decompute"]["models"].as_array().unwrap();
        for model in pi_models {
            assert!(model["contextWindow"].as_u64().unwrap() == 32_768);
        }
    }
}
