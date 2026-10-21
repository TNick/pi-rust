//! Reconciliation of configured and runtime provider model rows.

use std::io::Write;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use rpi_ai::Model;
use thiserror::Error;

use crate::ModelCatalog;

/// Runtime and optional host-configured model rows for one provider.
#[derive(Debug, Clone)]
pub struct ProviderModelRows {
    /// Stable host-selected provider identity.
    pub provider_id: String,
    /// Runtime or built-in rows restored when no host override is configured.
    pub baseline_models: Vec<Model>,
    /// Configured rows that replace the baseline, including an empty override.
    pub configured_models: Option<Vec<Model>>,
}

impl ProviderModelRows {
    /// Creates provider rows with an optional configuration override.
    ///
    /// # Parameters
    /// - `provider_id`: Stable identity shared by the configured and baseline rows.
    /// - `baseline_models`: Runtime or built-in rows to restore when unconfigured.
    /// - `configured_models`: Optional host-configured replacement rows.
    ///
    /// # Returns
    /// A source description for one provider's model catalog rows.
    pub fn new(
        provider_id: impl Into<String>,
        baseline_models: Vec<Model>,
        configured_models: Option<Vec<Model>>,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            baseline_models,
            configured_models,
        }
    }
}

impl ModelCatalog {
    /// Reconciles a complete ordered set of configured and runtime model rows.
    ///
    /// Configured rows replace each provider's baseline. Providers with no
    /// configured rows use their baseline, and empty results are omitted. The
    /// catalog publishes one coherent snapshot update.
    ///
    /// # Parameters
    /// - `providers`: Provider sources in desired catalog order.
    ///
    /// # Returns
    /// The catalog revision after reconciliation.
    pub fn reconcile_provider_rows(
        &self,
        providers: impl IntoIterator<Item = ProviderModelRows>,
    ) -> u64 {
        self.reconcile_provider_rows_snapshot(providers).revision
    }

    /// Reconciles provider rows and returns the exact published snapshot.
    ///
    /// Configured rows replace each provider's baseline. Providers with no
    /// configured rows use their baseline, and empty results are omitted. The
    /// returned snapshot is captured atomically with the published update.
    ///
    /// # Parameters
    /// - `providers`: Provider sources in desired catalog order.
    ///
    /// # Returns
    /// The exact coherent catalog snapshot committed by this reconciliation.
    pub fn reconcile_provider_rows_snapshot(
        &self,
        providers: impl IntoIterator<Item = ProviderModelRows>,
    ) -> crate::ModelCatalogSnapshot {
        let mut ordered = IndexMap::new();
        for provider in providers {
            let models = provider
                .configured_models
                .unwrap_or(provider.baseline_models);
            if !models.is_empty() {
                ordered.insert(provider.provider_id, models);
            }
        }
        self.reconcile_ordered_providers_snapshot(ordered)
    }
}

/// A failure while loading or atomically saving provider model configuration.
#[derive(Debug, Error)]
pub enum ModelsConfigFileError {
    /// The selected file could not be read.
    #[error("could not read {path}: {source}")]
    Read {
        /// Path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The selected file contained invalid JSON.
    #[error("invalid JSON in {path}: {source}")]
    Json {
        /// Path that failed.
        path: PathBuf,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// The selected file could not be written.
    #[error("could not write {path}: {source}")]
    Write {
        /// Path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Loads or atomically updates one host-selected models configuration file.
pub struct ModelsConfigFile;

impl ModelsConfigFile {
    /// Loads a JSON configuration, using an empty configuration when absent.
    ///
    /// `//` comments outside quoted strings are accepted for compatibility
    /// with hand-edited provider files.
    ///
    /// # Parameters
    /// - `path`: Exact host-selected configuration path.
    ///
    /// # Returns
    /// Parsed configuration or an empty configuration if the file is absent.
    ///
    /// # Errors
    /// Returns an error when the file cannot be read or parsed.
    pub fn load(path: &Path) -> Result<crate::ModelsConfig, ModelsConfigFileError> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(crate::ModelsConfig::default());
            }
            Err(source) => {
                return Err(ModelsConfigFileError::Read {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        let parsed = serde_json::from_str(&text)
            .or_else(|first| serde_json::from_str(&strip_line_comments(&text)).map_err(|_| first));
        parsed.map_err(|source| ModelsConfigFileError::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Atomically replaces one host-selected file with modeled configuration.
    ///
    /// Unknown fields and comments are not preserved during a rewrite.
    ///
    /// # Parameters
    /// - `path`: Exact host-selected configuration path.
    /// - `config`: Configuration to serialize.
    ///
    /// # Errors
    /// Returns an error if serialization, staging, syncing, or publication fails.
    pub fn save(path: &Path, config: &crate::ModelsConfig) -> Result<(), ModelsConfigFileError> {
        let text =
            serde_json::to_vec_pretty(config).map_err(|source| ModelsConfigFileError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        atomic_write(path, &text)
    }

    /// Loads, mutates, and atomically publishes one configuration file.
    ///
    /// # Parameters
    /// - `path`: Exact host-selected configuration path.
    /// - `update`: Mutation applied to the loaded configuration.
    ///
    /// # Errors
    /// Returns an error if loading or atomic publication fails.
    pub fn update(
        path: &Path,
        update: impl FnOnce(&mut crate::ModelsConfig),
    ) -> Result<(), ModelsConfigFileError> {
        let mut config = Self::load(path)?;
        update(&mut config);
        Self::save(path, &config)
    }
}

/// Writes bytes through a synced sibling staging file and atomic rename.
///
/// # Parameters
/// - `path`: Destination path.
/// - `bytes`: Serialized file contents.
///
/// # Returns
/// Success after the destination is atomically replaced.
///
/// # Errors
/// Returns an error when staging or publishing fails.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ModelsConfigFileError> {
    let parent = path.parent().ok_or_else(|| ModelsConfigFileError::Write {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no parent"),
    })?;
    std::fs::create_dir_all(parent).map_err(|source| ModelsConfigFileError::Write {
        path: parent.to_path_buf(),
        source,
    })?;
    let temp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("models"),
        uuid::Uuid::new_v4().simple()
    ));
    let result = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temp)
            .map_err(|source| ModelsConfigFileError::Write {
                path: temp.clone(),
                source,
            })?;
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|source| ModelsConfigFileError::Write {
                path: temp.clone(),
                source,
            })?;
        drop(file);
        std::fs::rename(&temp, path).map_err(|source| ModelsConfigFileError::Write {
            path: path.to_path_buf(),
            source,
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

/// Removes `//` comments outside JSON quoted strings.
fn strip_line_comments(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut escaped = false;
    while let Some(ch) = chars.next() {
        if quoted {
            result.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                quoted = false;
            }
        } else if ch == '"' {
            quoted = true;
            result.push(ch);
        } else if ch == '/' && chars.peek() == Some(&'/') {
            for comment_char in chars.by_ref() {
                if comment_char == '\n' {
                    result.push('\n');
                    break;
                }
            }
        } else {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    //! Tests configured and baseline provider row reconciliation.

    use crate::{ModelCatalog, ProviderModelRows};
    use rpi_ai::{Api, Model};

    /// Builds a model row for provider catalog reconciliation.
    fn model(provider: &str, id: &str) -> Model {
        Model::new(
            id,
            id,
            Api::OpenaiCompletions,
            provider,
            "https://example.test",
        )
    }

    /// Configured rows replace defaults and missing config restores defaults.
    #[test]
    fn configured_rows_override_and_baselines_restore() {
        let catalog = ModelCatalog::new();
        catalog.reconcile_provider_rows([
            ProviderModelRows::new(
                "builtin",
                vec![model("builtin", "default")],
                Some(vec![model("builtin", "configured")]),
            ),
            ProviderModelRows::new("extension", vec![model("extension", "runtime")], None),
        ]);
        assert_eq!(
            catalog
                .snapshot()
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["configured", "runtime"]
        );

        catalog.reconcile_provider_rows([
            ProviderModelRows::new("builtin", vec![model("builtin", "default")], None),
            ProviderModelRows::new("extension", vec![model("extension", "runtime")], None),
        ]);
        assert_eq!(catalog.provider_models("builtin")[0].id, "default");
    }

    /// Empty configured rows remove overrides and do not expose baseline rows.
    #[test]
    fn empty_configured_rows_suppress_baseline_rows() {
        let catalog = ModelCatalog::new();
        catalog.reconcile_provider_rows([ProviderModelRows::new(
            "provider",
            vec![model("provider", "default")],
            Some(Vec::new()),
        )]);
        assert!(catalog.snapshot().models.is_empty());
    }

    /// Reconciles provider rows in one catalog revision and event.
    #[test]
    fn provider_row_reconciliation_returns_published_snapshot() {
        let catalog = ModelCatalog::new();
        let mut events = catalog.subscribe();
        let snapshot = catalog.reconcile_provider_rows_snapshot([
            ProviderModelRows::new("first", vec![model("first", "a")], None),
            ProviderModelRows::new("second", vec![model("second", "b")], None),
        ]);
        let event = events.try_recv().expect("one catalog update");
        assert_eq!(snapshot, catalog.snapshot());
        assert_eq!(snapshot.revision, event.revision);
        assert_eq!(event.provider_ids, ["first", "second"]);
        assert!(events.try_recv().is_err());
    }

    /// Reads, updates, and reloads a path-selected models configuration.
    #[test]
    fn models_config_file_updates_path_selected_configuration() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("models.json");
        let result = super::ModelsConfigFile::update(&path, |config| {
            config.providers.insert(
                "gateway".into(),
                crate::ProviderConfig {
                    name: Some("Gateway".into()),
                    base_url: None,
                    api_key: None,
                    api: Some("openai-completions".into()),
                    headers: None,
                    auth_header: None,
                    models: Vec::new(),
                },
            );
        });
        assert!(result.is_ok());
        let loaded = super::ModelsConfigFile::load(&path).unwrap();
        assert_eq!(loaded.providers["gateway"].name.as_deref(), Some("Gateway"));
    }

    /// Loads absent configuration as empty and accepts line comments.
    #[test]
    fn models_config_file_loads_missing_and_commented_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("models.json");
        assert!(super::ModelsConfigFile::load(&path)
            .unwrap()
            .providers
            .is_empty());
        std::fs::write(&path, "{\n // provider config\n \"providers\": {}\n}").unwrap();
        assert!(super::ModelsConfigFile::load(&path)
            .unwrap()
            .providers
            .is_empty());
    }
}
