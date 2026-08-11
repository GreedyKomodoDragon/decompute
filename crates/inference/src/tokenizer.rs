use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub fn load(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|err| anyhow::anyhow!(err))
        .context("load tokenizer.json")
}
