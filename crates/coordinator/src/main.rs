mod api;
mod registry;
mod scheduler;

use anyhow::Result;
use clap::Parser;
use registry::Registry;
use std::{net::SocketAddr, sync::Arc};
use tokio::{
    net::TcpListener,
    time::{Duration, interval},
};
use tower_http::trace::TraceLayer;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();
    let registry = Arc::new(Registry::default());
    let expiry_registry = registry.clone();
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(5));
        loop {
            ticker.tick().await;
            expiry_registry.expire_stale(Duration::from_secs(15)).await;
        }
    });
    let app = api::router(registry).layer(TraceLayer::new_for_http());
    let bind: SocketAddr = format!("{}:{}", args.bind, args.port).parse()?;
    info!(%bind, "coordinator listening");
    axum::serve(TcpListener::bind(bind).await?, app).await?;
    Ok(())
}
