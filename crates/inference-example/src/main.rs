use anyhow::Result;
use inference::{GenerationConfig, LocalModel};

fn main() -> Result<()> {
    let mut model = LocalModel::load("./models/tiny-model")?;
    let result = model.generate("Why is the sky blue?", GenerationConfig::default())?;
    println!("{}", result.text);
    Ok(())
}
