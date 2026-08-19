//! Curated model source resolution and verification.
//!
//! This crate owns model acquisition, not inference. Consumers resolve a
//! catalog entry into a verified local file and pass that file to the runtime
//! SDK. The coordinator and OpenAI-compatible clients stay source agnostic.

use anyhow::{Context, Result, bail};
use decompute_core::{ModelFile, ModelManifest};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

const EMBEDDED_CATALOG: &str = include_str!("../catalog.toml");

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct ModelCatalog {
    models: Vec<CatalogModel>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct CatalogModel {
    pub id: String,
    pub display_name: String,
    pub format: String,
    pub repository: String,
    pub revision: String,
    pub filename: String,
    pub sha256: String,
    pub architecture: String,
    pub quantization: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedModel {
    pub entry: CatalogModel,
    pub path: PathBuf,
    pub manifest: ModelManifest,
    pub source: ModelSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSource {
    HuggingFaceCache,
    LocalOverride,
}

impl ModelCatalog {
    pub fn embedded() -> Result<Self> {
        let catalog: Self =
            toml::from_str(EMBEDDED_CATALOG).context("parse embedded model catalog")?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn from_toml(input: &str) -> Result<Self> {
        let catalog: Self = toml::from_str(input).context("parse model catalog")?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn get(&self, id: &str) -> Result<&CatalogModel> {
        self.models
            .iter()
            .find(|model| model.id == id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown curated model `{id}`; available models: {}",
                    self.models
                        .iter()
                        .map(|model| model.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }

    pub fn entries(&self) -> &[CatalogModel] {
        &self.models
    }

    fn validate(&self) -> Result<()> {
        if self.models.is_empty() {
            bail!("model catalog contains no models");
        }
        for model in &self.models {
            validate_model(model)?;
            if self
                .models
                .iter()
                .filter(|other| other.id == model.id)
                .count()
                != 1
            {
                bail!("model catalog contains duplicate id `{}`", model.id);
            }
        }
        Ok(())
    }
}

/// Resolves a curated model to a verified local GGUF file. An explicit local
/// override remains useful for offline use and shares the catalog's integrity
/// checks. Without an override, the Hugging Face cache is used automatically.
pub async fn resolve(model: CatalogModel, local_override: Option<&Path>) -> Result<ResolvedModel> {
    validate_model(&model)?;
    let (path, source) = match local_override {
        Some(path) => (path.to_path_buf(), ModelSource::LocalOverride),
        None => (
            download_from_hugging_face(&model).await?,
            ModelSource::HuggingFaceCache,
        ),
    };
    verify_file(&path, &model).with_context(|| format!("verify curated model `{}`", model.id))?;
    let manifest = manifest_for(&path, &model);
    Ok(ResolvedModel {
        entry: model,
        path,
        manifest,
        source,
    })
}

async fn download_from_hugging_face(model: &CatalogModel) -> Result<PathBuf> {
    let (owner, name) = model.repository.split_once('/').ok_or_else(|| {
        anyhow::anyhow!(
            "invalid Hugging Face repository `{}`; expected owner/name",
            model.repository
        )
    })?;
    let client = hf_hub::HFClient::builder()
        .user_agent(format!("decompute-models/{}", env!("CARGO_PKG_VERSION")));
    let client = client.build().context("create Hugging Face client")?;
    client
        .model(owner, name)
        .download_file()
        .filename(&model.filename)
        .revision(&model.revision)
        .send()
        .await
        .with_context(|| {
            format!(
                "download {} at revision {} from Hugging Face",
                model.repository, model.revision
            )
        })
}

fn validate_model(model: &CatalogModel) -> Result<()> {
    for (field, value) in [
        ("id", model.id.as_str()),
        ("display_name", model.display_name.as_str()),
        ("repository", model.repository.as_str()),
        ("revision", model.revision.as_str()),
        ("filename", model.filename.as_str()),
        ("architecture", model.architecture.as_str()),
    ] {
        if value.trim().is_empty() {
            bail!("model catalog entry has an empty `{field}` field");
        }
    }
    let Some((owner, name)) = model.repository.split_once('/') else {
        bail!(
            "model `{}` has invalid Hugging Face repository `{}`; expected owner/name",
            model.id,
            model.repository
        );
    };
    if owner.trim().is_empty() || name.trim().is_empty() || name.contains('/') {
        bail!(
            "model `{}` has invalid Hugging Face repository `{}`; expected owner/name",
            model.id,
            model.repository
        );
    }
    if model.format != "gguf" {
        bail!(
            "model `{}` has unsupported format `{}`; only gguf is supported",
            model.id,
            model.format
        );
    }
    if Path::new(&model.filename)
        .extension()
        .and_then(|extension| extension.to_str())
        != Some("gguf")
    {
        bail!(
            "model `{}` has filename `{}` which does not match gguf format",
            model.id,
            model.filename
        );
    }
    if !is_sha256(&model.sha256) {
        bail!("model `{}` has an invalid sha256 digest", model.id);
    }
    Ok(())
}

fn verify_file(path: &Path, model: &CatalogModel) -> Result<()> {
    if !path.is_file() {
        bail!("model file does not exist: {}", path.display());
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        bail!("expected a .gguf model file, got {}", path.display());
    }
    let digest = sha256_file(path)?;
    if !digest.eq_ignore_ascii_case(&model.sha256) {
        bail!(
            "SHA-256 mismatch for {}: expected {}, got {}",
            path.display(),
            model.sha256,
            digest
        );
    }
    Ok(())
}

fn manifest_for(path: &Path, model: &CatalogModel) -> ModelManifest {
    ModelManifest {
        id: format!("sha256:{}", model.sha256),
        architecture: model.architecture.clone(),
        revision: model.revision.clone(),
        quantization: model.quantization.clone(),
        files: vec![ModelFile {
            path: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned(),
            sha256: model.sha256.clone(),
        }],
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn embedded_catalog_is_valid_and_contains_qwen() {
        let catalog = ModelCatalog::embedded().unwrap();
        assert_eq!(
            catalog
                .get("qwen2.5-0.5b-instruct-q4-k-m")
                .unwrap()
                .repository,
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        );
        assert_eq!(
            catalog
                .get("qwen2.5-1.5b-instruct-q4-k-m")
                .unwrap()
                .revision,
            "91cad51170dc346986eccefdc2dd33a9da36ead9"
        );
    }

    #[test]
    fn invalid_catalog_digest_is_rejected() {
        let source = EMBEDDED_CATALOG.replace(
            "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
            "not-a-digest",
        );
        assert!(ModelCatalog::from_toml(&source).is_err());
    }

    #[test]
    fn catalog_rejects_missing_provenance_fields_and_malformed_repositories() {
        for (from, to) in [
            (
                "revision = \"872f8a96064a1242ac3a3359cad77c3042548405\"",
                "revision = \"\"",
            ),
            ("architecture = \"qwen2\"", "architecture = \"\""),
            (
                "display_name = \"Qwen2.5 0.5B Instruct Q4_K_M\"",
                "display_name = \"\"",
            ),
            (
                "repository = \"Qwen/Qwen2.5-0.5B-Instruct-GGUF\"",
                "repository = \"Qwen/invalid/extra\"",
            ),
            (
                "filename = \"qwen2.5-0.5b-instruct-q4_k_m.gguf\"",
                "filename = \"qwen2.5-0.5b-instruct-q4_k_m.bin\"",
            ),
        ] {
            let source = EMBEDDED_CATALOG.replacen(from, to, 1);
            assert!(ModelCatalog::from_toml(&source).is_err(), "{from}");
        }
    }

    #[test]
    fn catalog_rejects_duplicate_ids() {
        let duplicate = EMBEDDED_CATALOG.replacen(
            "id = \"qwen2.5-1.5b-instruct-q4-k-m\"",
            "id = \"qwen2.5-0.5b-instruct-q4-k-m\"",
            1,
        );
        assert!(ModelCatalog::from_toml(&duplicate).is_err());
    }

    #[test]
    fn local_override_must_match_catalog_digest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"not a model").unwrap();
        let model = ModelCatalog::embedded()
            .unwrap()
            .get("qwen2.5-0.5b-instruct-q4-k-m")
            .unwrap()
            .clone();
        assert!(verify_file(&path, &model).is_err());
    }

    #[test]
    fn verified_local_file_uses_deterministic_manifest_identity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let contents = b"verified test model";
        std::fs::write(&path, contents).unwrap();
        let mut model = ModelCatalog::embedded()
            .unwrap()
            .get("qwen2.5-0.5b-instruct-q4-k-m")
            .unwrap()
            .clone();
        model.sha256 = format!("{:x}", Sha256::digest(contents));

        verify_file(&path, &model).unwrap();
        assert_eq!(
            manifest_for(&path, &model).id,
            format!("sha256:{}", model.sha256)
        );
    }
}
