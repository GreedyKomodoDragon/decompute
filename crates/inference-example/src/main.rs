use anyhow::Result;
use inference::{GenerationConfig, LocalModel};
use protocol::{ChatMessage, ChatRole};

fn main() -> Result<()> {
    let mut model = LocalModel::load("./models/tiny-model")?;
    let result = model.generate(
        &[ChatMessage {
            role: ChatRole::User,
            content: "Why is the sky blue?".into(),
        }],
        None,
        GenerationConfig::default(),
    )?;
    println!("{}", result.text);
    Ok(())
}
