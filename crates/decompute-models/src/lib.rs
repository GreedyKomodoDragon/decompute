//! Curated model source resolution and verification.
//!
//! This crate owns model acquisition, not inference. Consumers resolve a
//! catalog entry into a verified local file and pass that file to the runtime
//! SDK. The coordinator and OpenAI-compatible clients stay source agnostic.

use anyhow::{Context, Result, bail};
use decompute_core::{ModelFile, ModelManifest};
use hf_hub::{
    HFError,
    progress::{DownloadEvent, ProgressEvent, ProgressHandler},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

const EMBEDDED_CATALOG: &str = include_str!("../catalog.toml");

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct ModelCatalog {
    models: Vec<CatalogModel>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct CatalogModel {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub download_size_bytes: u64,
    pub recommended: bool,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelCacheStatus {
    Missing,
    Verified(Box<ResolvedModel>),
    Corrupt { path: PathBuf, error: String },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelResolveStage {
    CheckingCache,
    Downloading {
        bytes_downloaded: u64,
        total_bytes: u64,
        bytes_per_sec: Option<f64>,
    },
    Verifying {
        bytes_verified: u64,
        total_bytes: u64,
    },
    Ready,
}

pub trait ModelProgressObserver: Send + Sync {
    fn on_progress(&self, stage: ModelResolveStage);
}

impl<F> ModelProgressObserver for F
where
    F: Fn(ModelResolveStage) + Send + Sync,
{
    fn on_progress(&self, stage: ModelResolveStage) {
        self(stage);
    }
}

#[derive(Debug, Error)]
pub enum ModelResolveError {
    #[error("model resolution was cancelled")]
    Cancelled,
    #[error(
        "not enough free space in model cache: {available_bytes} bytes available, {required_bytes} bytes required"
    )]
    InsufficientDiskSpace {
        available_bytes: u64,
        required_bytes: u64,
    },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
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
        if self.models.iter().filter(|model| model.recommended).count() != 1 {
            bail!("model catalog must contain exactly one recommended model");
        }
        Ok(())
    }
}

/// Resolves a curated model to a verified local GGUF file. An explicit local
/// override remains useful for offline use and shares the catalog's integrity
/// checks. Without an override, the Hugging Face cache is used automatically.
pub async fn resolve(model: CatalogModel, local_override: Option<&Path>) -> Result<ResolvedModel> {
    resolve_with_progress(
        model,
        local_override,
        Arc::new(|_: ModelResolveStage| {}),
        CancellationToken::new(),
    )
    .await
    .map_err(anyhow::Error::new)
}

pub fn cache_dir() -> Result<PathBuf> {
    Ok(hugging_face_client()?.cache_dir().to_path_buf())
}

pub async fn inspect_cache(model: CatalogModel) -> Result<ModelCacheStatus> {
    validate_model(&model)?;
    let repository = hugging_face_repository(&model)?;
    let path = match repository
        .download_file()
        .filename(&model.filename)
        .revision(&model.revision)
        .local_files_only(true)
        .send()
        .await
    {
        Ok(path) => path,
        Err(HFError::LocalEntryNotFound { .. }) => return Ok(ModelCacheStatus::Missing),
        Err(error) => return Err(error).context("inspect Hugging Face model cache"),
    };
    let verify_path = path.clone();
    let verify_model = model.clone();
    match tokio::task::spawn_blocking(move || verify_file(&verify_path, &verify_model)).await {
        Ok(Ok(())) => Ok(ModelCacheStatus::Verified(Box::new(resolved_model(
            model,
            path,
            ModelSource::HuggingFaceCache,
        )))),
        Ok(Err(error)) => Ok(ModelCacheStatus::Corrupt {
            path,
            error: error.to_string(),
        }),
        Err(error) => Err(error).context("join model cache verification task"),
    }
}

pub async fn resolve_with_progress(
    model: CatalogModel,
    local_override: Option<&Path>,
    observer: Arc<dyn ModelProgressObserver>,
    cancellation: CancellationToken,
) -> std::result::Result<ResolvedModel, ModelResolveError> {
    validate_model(&model)?;
    observer.on_progress(ModelResolveStage::CheckingCache);
    if cancellation.is_cancelled() {
        return Err(ModelResolveError::Cancelled);
    }
    let (path, source) = match local_override {
        Some(path) => (path.to_path_buf(), ModelSource::LocalOverride),
        None => {
            let force_download = match inspect_cache(model.clone()).await? {
                ModelCacheStatus::Verified(resolved) => {
                    observer.on_progress(ModelResolveStage::Ready);
                    return Ok(*resolved);
                }
                ModelCacheStatus::Corrupt { .. } => true,
                ModelCacheStatus::Missing => false,
            };
            preflight_disk_space(&model)?;
            (
                download_from_hugging_face(
                    &model,
                    observer.clone(),
                    cancellation.clone(),
                    force_download,
                )
                .await?,
                ModelSource::HuggingFaceCache,
            )
        }
    };
    verify_file_async(&path, &model, observer.clone(), cancellation).await?;
    observer.on_progress(ModelResolveStage::Ready);
    Ok(resolved_model(model, path, source))
}

async fn download_from_hugging_face(
    model: &CatalogModel,
    observer: Arc<dyn ModelProgressObserver>,
    cancellation: CancellationToken,
    force_download: bool,
) -> std::result::Result<PathBuf, ModelResolveError> {
    let repository = hugging_face_repository(model)?;
    let progress = DownloadProgress { observer };
    tokio::select! {
        () = cancellation.cancelled() => Err(ModelResolveError::Cancelled),
        result = repository
            .download_file()
            .filename(&model.filename)
            .revision(&model.revision)
            .force_download(force_download)
            .progress(progress)
            .send() => result.with_context(|| {
                format!(
                    "download {} at revision {} from Hugging Face",
                    model.repository, model.revision
                )
            }).map_err(ModelResolveError::Other),
    }
}

fn hugging_face_client() -> Result<hf_hub::HFClient> {
    hf_hub::HFClient::builder()
        .user_agent(format!("decompute-models/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .context("create Hugging Face client")
}

fn hugging_face_repository(
    model: &CatalogModel,
) -> Result<hf_hub::HFRepository<hf_hub::RepoTypeModel>> {
    let (owner, name) = model.repository.split_once('/').ok_or_else(|| {
        anyhow::anyhow!(
            "invalid Hugging Face repository `{}`; expected owner/name",
            model.repository
        )
    })?;
    Ok(hugging_face_client()?.model(owner, name))
}

struct DownloadProgress {
    observer: Arc<dyn ModelProgressObserver>,
}

impl ProgressHandler for DownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let stage = match event {
            DownloadEvent::Start { total_bytes, .. } => Some(ModelResolveStage::Downloading {
                bytes_downloaded: 0,
                total_bytes: *total_bytes,
                bytes_per_sec: None,
            }),
            DownloadEvent::Progress { files } => {
                files.last().map(|file| ModelResolveStage::Downloading {
                    bytes_downloaded: file.bytes_completed,
                    total_bytes: file.total_bytes,
                    bytes_per_sec: None,
                })
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            } => Some(ModelResolveStage::Downloading {
                bytes_downloaded: *bytes_completed,
                total_bytes: *total_bytes,
                bytes_per_sec: *bytes_per_sec,
            }),
            DownloadEvent::Complete => None,
        };
        if let Some(stage) = stage {
            self.observer.on_progress(stage);
        }
    }
}

fn preflight_disk_space(model: &CatalogModel) -> std::result::Result<(), ModelResolveError> {
    let cache = cache_dir()?;
    std::fs::create_dir_all(&cache)
        .with_context(|| format!("create model cache directory {}", cache.display()))?;
    let available = fs2::available_space(&cache)
        .with_context(|| format!("inspect free space for {}", cache.display()))?;
    let reserve = (model.download_size_bytes / 10).max(256 * 1024 * 1024);
    let required = model.download_size_bytes.saturating_add(reserve);
    if available < required {
        return Err(ModelResolveError::InsufficientDiskSpace {
            available_bytes: available,
            required_bytes: required,
        });
    }
    Ok(())
}

async fn verify_file_async(
    path: &Path,
    model: &CatalogModel,
    observer: Arc<dyn ModelProgressObserver>,
    cancellation: CancellationToken,
) -> std::result::Result<(), ModelResolveError> {
    let path = path.to_path_buf();
    let model = model.clone();
    tokio::task::spawn_blocking(move || {
        verify_file_with_progress(&path, &model, observer.as_ref(), &cancellation)
    })
    .await
    .context("join model verification task")??;
    Ok(())
}

fn resolved_model(model: CatalogModel, path: PathBuf, source: ModelSource) -> ResolvedModel {
    let manifest = manifest_for(&path, &model);
    ResolvedModel {
        entry: model,
        path,
        manifest,
        source,
    }
}

fn validate_model(model: &CatalogModel) -> Result<()> {
    for (field, value) in [
        ("id", model.id.as_str()),
        ("display_name", model.display_name.as_str()),
        ("description", model.description.as_str()),
        ("repository", model.repository.as_str()),
        ("revision", model.revision.as_str()),
        ("filename", model.filename.as_str()),
        ("architecture", model.architecture.as_str()),
    ] {
        if value.trim().is_empty() {
            bail!("model catalog entry has an empty `{field}` field");
        }
    }
    if model.download_size_bytes == 0 {
        bail!("model `{}` has a zero download size", model.id);
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
    Ok(verify_file_with_progress(
        path,
        model,
        &|_: ModelResolveStage| {},
        &CancellationToken::new(),
    )?)
}

fn verify_file_with_progress(
    path: &Path,
    model: &CatalogModel,
    observer: &dyn ModelProgressObserver,
    cancellation: &CancellationToken,
) -> std::result::Result<(), ModelResolveError> {
    if !path.is_file() {
        return Err(anyhow::anyhow!("model file does not exist: {}", path.display()).into());
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return Err(anyhow::anyhow!("expected a .gguf model file, got {}", path.display()).into());
    }
    let total = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    let digest = sha256_file_with_progress(path, total, observer, cancellation)?;
    if !digest.eq_ignore_ascii_case(&model.sha256) {
        return Err(anyhow::anyhow!(
            "SHA-256 mismatch for {}: expected {}, got {}",
            path.display(),
            model.sha256,
            digest
        )
        .into());
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

fn sha256_file_with_progress(
    path: &Path,
    total: u64,
    observer: &dyn ModelProgressObserver,
    cancellation: &CancellationToken,
) -> std::result::Result<String, ModelResolveError> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut verified = 0_u64;
    loop {
        if cancellation.is_cancelled() {
            return Err(ModelResolveError::Cancelled);
        }
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        verified = verified.saturating_add(read as u64);
        observer.on_progress(ModelResolveStage::Verifying {
            bytes_verified: verified,
            total_bytes: total,
        });
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
                .entries()
                .iter()
                .filter(|model| model.recommended)
                .count(),
            1
        );
        assert!(
            catalog
                .entries()
                .iter()
                .all(|model| model.download_size_bytes > 0)
        );
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
    fn catalog_requires_exactly_one_recommended_model() {
        let none = EMBEDDED_CATALOG.replace("recommended = true", "recommended = false");
        assert!(ModelCatalog::from_toml(&none).is_err());
        let both = EMBEDDED_CATALOG.replace("recommended = false", "recommended = true");
        assert!(ModelCatalog::from_toml(&both).is_err());
    }

    #[tokio::test]
    async fn cancelled_local_resolution_never_reports_ready() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let contents = b"verified test model";
        std::fs::write(&path, contents).unwrap();
        let mut model = ModelCatalog::embedded().unwrap().entries()[0].clone();
        model.sha256 = format!("{:x}", Sha256::digest(contents));
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = events.clone();
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let result = resolve_with_progress(
            model,
            Some(&path),
            Arc::new(move |stage| observed.lock().unwrap().push(stage)),
            cancellation,
        )
        .await;

        assert!(matches!(result, Err(ModelResolveError::Cancelled)));
        assert!(
            !events
                .lock()
                .unwrap()
                .iter()
                .any(|stage| matches!(stage, ModelResolveStage::Ready))
        );
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
