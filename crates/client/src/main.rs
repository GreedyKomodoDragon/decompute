use anyhow::Result;
use clap::Parser;
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    coordinator: String,
    #[arg(long, default_value = "qwen2.5-0.5b-instruct-q4-k-m")]
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
            "{}/v1/chat/completions",
            args.coordinator.trim_end_matches('/')
        ))
        .json(&json!({
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
        }))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
