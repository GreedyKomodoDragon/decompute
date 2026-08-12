use anyhow::Result;
use clap::Parser;
use protocol::{PublicGenerateRequest, PublicGenerateResponse};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    coordinator: String,
    #[arg(long, default_value = "tiny-model")]
    model: String,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,
}
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let response = reqwest::Client::new()
        .post(format!(
            "{}/v1/generate",
            args.coordinator.trim_end_matches('/')
        ))
        .json(&PublicGenerateRequest {
            request_id: None,
            model: args.model,
            prompt: Some(args.prompt),
            messages: None,
            template: None,
            max_tokens: args.max_tokens,
        })
        .send()
        .await?
        .error_for_status()?
        .json::<PublicGenerateResponse>()
        .await?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
