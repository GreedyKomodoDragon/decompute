use anyhow::Result;
use inference::{GenerationConfig, LocalModel};
use protocol::{ChatMessage, ChatRole};
use uuid::Uuid;

fn main() -> Result<()> {
    let mut model = LocalModel::load("./models/tiny-model")?;
    let result = model.generate(
        &[ChatMessage {
            role: ChatRole::User,
            content: "Why is the sky blue?".into(),
            tool_calls: vec![],
            tool_call_id: None,
        }],
        None,
        &[],
        Uuid::new_v4(),
        GenerationConfig::default(),
    )?;
    println!("{}", result.text);
    Ok(())
}
