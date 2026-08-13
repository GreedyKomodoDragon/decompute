use anyhow::Result;
use decompute_sdk::{
    ChatMessage, ChatRequest, ChatRole, GenerationConfig, GgufLoadConfig, GgufModelHandle,
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    let model = GgufModelHandle::load("./models/tiny-model.gguf", GgufLoadConfig::default())?;
    let result = model
        .generate(ChatRequest {
            request_id: Uuid::new_v4(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "Why is the sky blue?".into(),
                tool_calls: vec![],
                tool_call_id: None,
            }],
            template: None,
            tools: vec![],
            generation: GenerationConfig::default(),
        })
        .await?;
    println!("{}", result.text);
    Ok(())
}
