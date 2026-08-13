mod api;
mod heartbeat;
mod resources;
mod state;

use anyhow::Result;
use axum::Router;
use clap::Parser;
use decompute_sdk::{GgufLoadConfig, GgufModelHandle};
use protocol::Acceleration;
use state::WorkerRuntime;
use std::{net::SocketAddr, sync::Arc};
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
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "tiny-model")]
    model_id: String,
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    #[arg(long)]
    advertise_address: Option<String>,
    #[arg(long, default_value_t = 1)]
    max_requests: usize,
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    device: DeviceChoice,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum DeviceChoice {
    Auto,
    Cpu,
    Metal,
}

async fn load_model(args: &Args) -> Result<(GgufModelHandle, Acceleration)> {
    match args.device {
        DeviceChoice::Cpu => {
            load_and_verify(&args.model, GgufLoadConfig::default(), Acceleration::Cpu).await
        }
        DeviceChoice::Metal => load_metal(&args.model).await,
        DeviceChoice::Auto => {
            #[cfg(feature = "metal")]
            {
                match load_and_verify(
                    &args.model,
                    GgufLoadConfig {
                        gpu_layers: Some(u32::MAX),
                        ..Default::default()
                    },
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
            load_and_verify(&args.model, GgufLoadConfig::default(), Acceleration::Cpu).await
        }
    }
}

async fn load_and_verify(
    path: &str,
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
async fn load_metal(path: &str) -> Result<(GgufModelHandle, Acceleration)> {
    load_and_verify(
        path,
        GgufLoadConfig {
            gpu_layers: Some(u32::MAX),
            ..Default::default()
        },
        Acceleration::Metal,
    )
    .await
}
#[cfg(not(feature = "metal"))]
async fn load_metal(_path: &str) -> Result<(GgufModelHandle, Acceleration)> {
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
    info!(model = %args.model, "loading local model");
    let manifest = decompute_llama::local_manifest(&args.model)?;
    let (model, acceleration) = load_model(&args).await?;
    let runtime = Arc::new(WorkerRuntime::new(
        args.node_id,
        args.model_id,
        address,
        args.max_requests,
        manifest,
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
