# decompute-sdk

Async embedded GGUF inference for Rust applications.

Enable a runtime feature and load a local model:

```toml
[dependencies]
decompute-sdk = { version = "0.1", features = ["llama-metal"] }
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
tokio-util = "0.7"
uuid = { version = "1", features = ["v4"] }
```

```rust,no_run
use decompute_sdk::{
    ChatMessage, ChatRequest, ChatRole, GenerationConfig, GgufLoadConfig,
    GgufModelHandle,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = GgufModelHandle::load("./model.gguf", GgufLoadConfig::default())?;
    let result = model
        .generate(ChatRequest {
            request_id: Uuid::new_v4(),
            session_id: Some(Uuid::new_v4()),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Summarize this email.".into(),
                tool_calls: vec![],
                tool_call_id: None,
            }],
            template: None,
            tools: vec![],
            generation: GenerationConfig::default(),
            cancellation: CancellationToken::new(),
        })
        .await?;
    println!("{}", result.text);
    Ok(())
}
```

The application owns tool execution. The model can return structured tool
calls, but this SDK never runs shell commands, network requests, or application
tools on the caller's behalf.

Feature flags:

- `llama`: native CPU/runtime support.
- `llama-metal`: native runtime with Apple Silicon Metal support.

The SDK keeps the model on a dedicated execution thread so native inference
does not block the application's async executor.
