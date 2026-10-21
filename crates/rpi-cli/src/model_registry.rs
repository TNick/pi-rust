//! Connects CLI provider configuration to the shared live model catalog.
//!
//! The reusable catalog owns model rows, coherent snapshots, and catalog update
//! notifications. This module retains CLI provider registration, auth-status
//! compatibility, package integration, and filesystem-backed refresh policy.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::broadcast;

use rpi_ai::model::Model;
use rpi_ai::Provider;

/// Model registry event types
#[derive(Debug, Clone)]
pub enum ModelRegistryEvent {
    /// A new provider was registered
    ProviderRegistered { provider_id: String },
    /// A provider was unregistered
    ProviderUnregistered { provider_id: String },
    /// Model catalog was refreshed
    CatalogRefreshed,
    /// Provider auth status changed
    AuthStatusChanged {
        provider_id: String,
        authenticated: bool,
    },
}

/// Provider authentication status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthStatus {
    /// Not authenticated
    Unauthenticated,
    /// Authenticated with valid credentials
    Authenticated,
    /// Authentication expired/invalid
    Expired,
    /// Authentication status unknown
    Unknown,
}

/// Provider information in the registry
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Provider ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Whether this is a built-in provider
    pub builtin: bool,
    /// Authentication status
    pub auth_status: AuthStatus,
    /// Available models
    pub models: Vec<Model>,
}

/// CLI-owned provider metadata independent of upstream model rows.
#[derive(Debug, Clone)]
struct ProviderRecord {
    /// Provider identity.
    id: String,
    /// Provider display name.
    name: String,
    /// Display name supplied by registration before config overrides.
    baseline_name: String,
    /// Whether this provider is built in.
    builtin: bool,
    /// CLI authentication status.
    auth_status: AuthStatus,
    /// Model rows supplied when the provider was registered.
    baseline_models: Vec<Model>,
    /// Whether this record exists only because models.json declared it.
    configured_only: bool,
}

/// Model registry - manages providers and models at runtime
pub struct ModelRegistry {
    /// Provider metadata owned by the CLI.
    providers: Arc<RwLock<indexmap::IndexMap<String, ProviderRecord>>>,
    /// Provider instances (for actual API calls)
    provider_instances: Arc<RwLock<HashMap<String, Arc<dyn Provider>>>>,
    /// Shared live catalog owned by the reusable model runtime.
    catalog: rpi_model_runtime::ModelCatalog,
    /// Event broadcaster
    event_tx: broadcast::Sender<ModelRegistryEvent>,
    /// Models.json last refresh time
    last_refresh: Arc<RwLock<Option<std::time::Instant>>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self::with_catalog(rpi_model_runtime::ModelCatalog::new())
    }

    /// Create a CLI registry backed by a caller-owned shared model catalog.
    ///
    /// This lets CLI metadata and extension reloads observe the same catalog
    /// snapshots and update notifications without moving configuration or
    /// credential policy into the runtime crate.
    ///
    /// # Parameters
    /// - `catalog`: Shared model catalog handle supplied by the host.
    ///
    /// # Returns
    /// A registry whose model reads and refreshes use `catalog`.
    pub fn with_catalog(catalog: rpi_model_runtime::ModelCatalog) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        Self {
            providers: Arc::new(RwLock::new(indexmap::IndexMap::new())),
            provider_instances: Arc::new(RwLock::new(std::collections::HashMap::new())),
            catalog,
            event_tx,
            last_refresh: Arc::new(RwLock::new(None)),
        }
    }

    /// Register a built-in provider
    pub fn register_builtin_provider(
        &self,
        id: String,
        name: String,
        models: Vec<Model>,
    ) -> Result<(), RegistryError> {
        let record = ProviderRecord {
            id: id.clone(),
            name: name.clone(),
            baseline_name: name,
            builtin: true,
            auth_status: AuthStatus::Unknown,
            baseline_models: models.clone(),
            configured_only: false,
        };

        let mut providers = self
            .providers
            .write()
            .map_err(|_| RegistryError::LockPoisoned)?;
        providers.insert(id.clone(), record);
        self.catalog.replace_provider_models(id.clone(), models);

        let _ = self
            .event_tx
            .send(ModelRegistryEvent::ProviderRegistered { provider_id: id });

        Ok(())
    }

    /// Register an extension provider
    pub fn register_extension_provider(
        &self,
        provider: Arc<dyn Provider>,
    ) -> Result<(), RegistryError> {
        let id = provider.id().to_string();
        let models = provider.models().to_vec();

        let record = ProviderRecord {
            id: id.clone(),
            name: id.clone(), // Extension providers use id as name
            baseline_name: id.clone(),
            builtin: false,
            auth_status: AuthStatus::Unknown,
            baseline_models: models.clone(),
            configured_only: false,
        };

        let mut providers = self
            .providers
            .write()
            .map_err(|_| RegistryError::LockPoisoned)?;
        providers.insert(id.clone(), record);
        self.catalog.replace_provider_models(id.clone(), models);

        let mut instances = self
            .provider_instances
            .write()
            .map_err(|_| RegistryError::LockPoisoned)?;
        instances.insert(id.clone(), provider);

        let _ = self
            .event_tx
            .send(ModelRegistryEvent::ProviderRegistered { provider_id: id });

        Ok(())
    }

    /// Unregister a provider
    pub fn unregister_provider(&self, provider_id: &str) -> Result<(), RegistryError> {
        let mut providers = self
            .providers
            .write()
            .map_err(|_| RegistryError::LockPoisoned)?;
        let mut instances = self
            .provider_instances
            .write()
            .map_err(|_| RegistryError::LockPoisoned)?;

        if providers.shift_remove(provider_id).is_some() {
            instances.remove(provider_id);
            self.catalog.remove_provider(provider_id);
            let _ = self
                .event_tx
                .send(ModelRegistryEvent::ProviderUnregistered {
                    provider_id: provider_id.to_string(),
                });
            Ok(())
        } else {
            Err(RegistryError::ProviderNotFound(provider_id.to_string()))
        }
    }

    /// Get provider info
    pub fn get_provider(&self, provider_id: &str) -> Option<ProviderInfo> {
        let providers = self.providers.read().ok()?;
        providers.get(provider_id).map(|record| ProviderInfo {
            id: record.id.clone(),
            name: record.name.clone(),
            builtin: record.builtin,
            auth_status: record.auth_status.clone(),
            models: self.catalog.provider_models(provider_id),
        })
    }

    /// Get all providers
    pub fn list_providers(&self) -> Vec<ProviderInfo> {
        self.providers
            .read()
            .ok()
            .map(|providers| {
                providers
                    .values()
                    .map(|record| ProviderInfo {
                        id: record.id.clone(),
                        name: record.name.clone(),
                        builtin: record.builtin,
                        auth_status: record.auth_status.clone(),
                        models: self.catalog.provider_models(&record.id),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all models across all providers
    pub fn list_models(&self) -> Vec<Model> {
        self.catalog.snapshot().models
    }

    /// Get models for a specific provider
    pub fn list_provider_models(&self, provider_id: &str) -> Vec<Model> {
        self.catalog.provider_models(provider_id)
    }

    /// Update provider authentication status
    pub fn update_auth_status(
        &self,
        provider_id: &str,
        status: AuthStatus,
    ) -> Result<(), RegistryError> {
        let mut providers = self
            .providers
            .write()
            .map_err(|_| RegistryError::LockPoisoned)?;

        if let Some(info) = providers.get_mut(provider_id) {
            info.auth_status = status.clone();

            let authenticated = matches!(status, AuthStatus::Authenticated);
            let _ = self.event_tx.send(ModelRegistryEvent::AuthStatusChanged {
                provider_id: provider_id.to_string(),
                authenticated,
            });

            Ok(())
        } else {
            Err(RegistryError::ProviderNotFound(provider_id.to_string()))
        }
    }

    /// Get provider authentication status
    pub fn get_auth_status(&self, provider_id: &str) -> Option<AuthStatus> {
        let providers = self.providers.read().ok()?;
        providers
            .get(provider_id)
            .map(|info| info.auth_status.clone())
    }

    /// Refresh model catalog from models.json
    pub async fn refresh_catalog(&self) -> Result<(), RegistryError> {
        // Read models.json from config directory
        let config = crate::config::load_models_config()
            .map_err(|e| RegistryError::InvalidConfiguration(e.to_string()))?;

        let configured = config
            .providers
            .iter()
            .map(|(provider_id, provider_config)| {
                (
                    provider_id.clone(),
                    (
                        provider_config.name.clone(),
                        crate::config::provider_to_models(provider_id, provider_config)
                            .unwrap_or_default(),
                    ),
                )
            })
            .collect::<indexmap::IndexMap<_, _>>();
        self.apply_catalog_refresh(configured)?;

        // Update last refresh time
        let mut last_refresh = self
            .last_refresh
            .write()
            .map_err(|_| RegistryError::LockPoisoned)?;
        *last_refresh = Some(std::time::Instant::now());

        // Broadcast CatalogRefreshed event
        let _ = self.event_tx.send(ModelRegistryEvent::CatalogRefreshed);

        Ok(())
    }

    /// Reconciles a complete models.json provider snapshot atomically.
    ///
    /// Providers registered as built-ins or extension runtimes remain
    /// registered when absent from models.json. Their runtime model rows are
    /// preserved or restored from the provider instance. Records created only
    /// from models.json are removed when their configuration disappears.
    ///
    /// - `configured`: Complete configured provider map of optional display
    ///   names and resolved model rows, in source order.
    ///
    /// Unit after provider metadata and catalog rows are reconciled.
    ///
    /// Returns a lock error before publishing when registry state is poisoned.
    /// # Errors
    /// Returns `LockPoisoned` if provider metadata or instances cannot be read.
    fn apply_catalog_refresh(
        &self,
        configured: indexmap::IndexMap<String, (Option<String>, Vec<Model>)>,
    ) -> Result<(), RegistryError> {
        let mut records = self
            .providers
            .write()
            .map_err(|_| RegistryError::LockPoisoned)?;
        let instances = self
            .provider_instances
            .read()
            .map_err(|_| RegistryError::LockPoisoned)?;

        let mut unconfigured_records = std::mem::take(&mut *records);
        let mut next_records = indexmap::IndexMap::new();
        let mut model_rows = Vec::new();

        for (provider_id, (name, models)) in configured {
            let baseline_name = provider_id.clone();
            let mut record = unconfigured_records
                .shift_remove(&provider_id)
                .unwrap_or_else(|| ProviderRecord {
                    id: provider_id.clone(),
                    name: baseline_name.clone(),
                    baseline_name: baseline_name.clone(),
                    builtin: false,
                    auth_status: AuthStatus::Unknown,
                    baseline_models: Vec::new(),
                    configured_only: true,
                });
            record.name = name.unwrap_or_else(|| record.baseline_name.clone());
            model_rows.push(rpi_model_runtime::ProviderModelRows::new(
                provider_id.clone(),
                record.baseline_models.clone(),
                Some(models),
            ));
            next_records.insert(provider_id.clone(), record);
        }

        for (provider_id, mut record) in unconfigured_records {
            if record.configured_only && !instances.contains_key(&provider_id) {
                continue;
            }
            record.name = record.baseline_name.clone();
            let models = if let Some(provider) = instances.get(&provider_id) {
                record.configured_only = false;
                provider.models().to_vec()
            } else {
                record.baseline_models.clone()
            };
            model_rows.push(rpi_model_runtime::ProviderModelRows::new(
                provider_id.clone(),
                models,
                None,
            ));
            next_records.insert(provider_id, record);
        }

        *records = next_records;

        self.catalog.reconcile_provider_rows(model_rows);
        Ok(())
    }

    /// Get last catalog refresh time
    pub fn last_refresh_time(&self) -> Option<std::time::Instant> {
        self.last_refresh.read().ok().and_then(|t| *t)
    }

    /// Subscribe to registry events
    pub fn subscribe(&self) -> broadcast::Receiver<ModelRegistryEvent> {
        self.event_tx.subscribe()
    }

    /// Returns the shared upstream live model catalog for snapshot consumers.
    ///
    /// # Returns
    /// A cloneable handle to the registry's model catalog.
    pub fn model_catalog(&self) -> rpi_model_runtime::ModelCatalog {
        self.catalog.clone()
    }

    /// Get provider instance for API calls
    pub fn get_provider_instance(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
        let instances = self.provider_instances.read().ok()?;
        instances.get(provider_id).cloned()
    }

    /// Check if a provider is registered
    pub fn has_provider(&self, provider_id: &str) -> bool {
        let providers = match self.providers.read() {
            Ok(p) => p,
            Err(_) => return false,
        };
        providers.contains_key(provider_id)
    }

    /// Get provider count
    pub fn provider_count(&self) -> usize {
        self.providers.read().map(|p| p.len()).unwrap_or(0)
    }

    /// Get model count
    pub fn model_count(&self) -> usize {
        self.catalog.snapshot().models.len()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry errors
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Lock poisoned")]
    LockPoisoned,

    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    #[error("Provider already registered: {0}")]
    ProviderAlreadyRegistered(String),

    #[error("Invalid provider configuration: {0}")]
    InvalidConfiguration(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a provider-qualified model for catalog refresh fixtures.
    fn model(provider_id: &str, model_id: &str) -> Model {
        Model::new(
            model_id,
            model_id,
            rpi_ai::Api::OpenaiCompletions,
            provider_id,
            "https://example.test/v1",
        )
    }

    /// Isolates a models.json refresh test from process-global configuration.
    struct ModelsConfigEnvironment {
        /// Shared lock for tests that mutate process-global configuration.
        _guard: std::sync::MutexGuard<'static, ()>,
        /// Temporary config directory removed when the fixture is dropped.
        _directory: tempfile::TempDir,
        /// Previous config path restored when the fixture is dropped.
        previous: Option<std::ffi::OsString>,
    }

    impl ModelsConfigEnvironment {
        /// Creates an isolated `RPI_CODING_AGENT_DIR` with no models file.
        fn new() -> Self {
            let guard = crate::config::test_support::env_lock()
                .lock()
                .expect("configuration environment lock");
            let previous = std::env::var_os(crate::config::CONFIG_DIR_ENV);
            let directory = tempfile::tempdir().expect("temporary config directory");
            std::env::set_var(crate::config::CONFIG_DIR_ENV, directory.path());
            Self {
                _guard: guard,
                _directory: directory,
                previous,
            }
        }

        /// Replaces the fixture's complete models.json contents.
        fn write(&self, json: &str) {
            std::fs::write(crate::config::models_path().expect("models path"), json)
                .expect("write models config");
        }
    }

    impl Drop for ModelsConfigEnvironment {
        /// Restores the process-global models config path.
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                std::env::set_var(crate::config::CONFIG_DIR_ENV, previous);
            } else {
                std::env::remove_var(crate::config::CONFIG_DIR_ENV);
            }
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.provider_count(), 0);
        assert_eq!(registry.model_count(), 0);
    }

    #[test]
    fn test_provider_registration() {
        let registry = ModelRegistry::new();

        assert!(registry
            .register_builtin_provider("test".to_string(), "Test Provider".to_string(), vec![])
            .is_ok());
        assert_eq!(registry.provider_count(), 1);
        assert!(registry.has_provider("test"));
    }

    /// Verifies CLI model reads and notifications use the shared runtime catalog.
    #[test]
    fn model_rows_follow_shared_catalog_replacements() {
        let registry = ModelRegistry::new();
        let catalog = registry.model_catalog();
        let mut events = catalog.subscribe();
        let initial = Model::new(
            "initial",
            "Initial",
            rpi_ai::Api::OpenaiCompletions,
            "test",
            "https://example.test",
        );
        registry
            .register_builtin_provider("test".to_owned(), "Test".to_owned(), vec![initial])
            .expect("register provider");
        assert_eq!(events.try_recv().unwrap().provider_ids, ["test"]);

        let refreshed = Model::new(
            "refreshed",
            "Refreshed",
            rpi_ai::Api::OpenaiCompletions,
            "test",
            "https://example.test",
        );
        catalog.replace_provider_models("test", vec![refreshed]);

        assert_eq!(registry.list_models()[0].id, "refreshed");
        assert_eq!(registry.list_provider_models("test")[0].id, "refreshed");
        assert_eq!(
            registry.get_provider("test").unwrap().models[0].id,
            "refreshed"
        );
        assert_eq!(events.try_recv().unwrap().revision, 2);
    }

    /// Reconciles a complete config reload in one event and retains runtimes.
    #[tokio::test]
    async fn refresh_replaces_and_removes_rows_atomically() {
        let config = ModelsConfigEnvironment::new();
        let registry = ModelRegistry::new();
        registry
            .register_builtin_provider(
                "builtin".to_owned(),
                "Built in".to_owned(),
                vec![model("builtin", "builtin-base")],
            )
            .expect("register built-in provider");
        let runtime_provider = rpi_ai::providers::faux::FauxProvider::with_models(
            rpi_ai::providers::faux::FauxScript::default(),
            vec![model("faux", "faux-runtime")],
        );
        registry
            .register_extension_provider(runtime_provider)
            .expect("register extension provider");

        config.write(
            r#"{"providers":{
                "builtin":{"api":"openai-completions","baseUrl":"https://example.test/v1","models":[{"id":"builtin-old"}]},
                "faux":{"name":"Faux configured","api":"openai-completions","baseUrl":"https://example.test/v1","models":[{"id":"faux-old"}]},
                "stale-custom":{"api":"openai-completions","baseUrl":"https://example.test/v1","models":[{"id":"stale-model"}]}
            }}"#,
        );
        registry.refresh_catalog().await.expect("initial refresh");

        let catalog = registry.model_catalog();
        let mut updates = catalog.subscribe();
        config.write(
            r#"{"providers":{
                "new-custom":{"api":"openai-completions","baseUrl":"https://example.test/v1","models":[{"id":"new-model"}]},
                "builtin":{"name":"Built-in updated","api":"openai-completions","baseUrl":"https://example.test/v1","models":[{"id":"builtin-new"}]}
            }}"#,
        );
        registry
            .refresh_catalog()
            .await
            .expect("replacement refresh");

        let event = updates.recv().await.expect("one catalog event");
        let snapshot = catalog.snapshot();
        assert_eq!(event.revision, snapshot.revision);
        assert_eq!(
            event.provider_ids,
            ["builtin", "faux", "new-custom", "stale-custom"]
        );
        assert_eq!(
            snapshot
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["new-model", "builtin-new", "faux-runtime"]
        );
        let providers = registry.list_providers();
        assert_eq!(
            providers
                .iter()
                .map(|provider| provider.id.as_str())
                .collect::<Vec<_>>(),
            ["new-custom", "builtin", "faux"]
        );
        assert_eq!(providers[0].name, "new-custom");
        assert_eq!(providers[1].name, "Built-in updated");
        assert_eq!(providers[2].name, "faux");
        assert!(registry.get_provider("stale-custom").is_none());
        assert!(registry.get_provider_instance("faux").is_some());
        assert_eq!(
            registry.get_provider("faux").unwrap().models[0].id,
            "faux-runtime"
        );
        assert!(matches!(
            updates.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));

        config.write(
            r#"{"providers":{
                "new-custom":{"api":"openai-completions","baseUrl":"https://example.test/v1","models":[{"id":"new-model"}]}
            }}"#,
        );
        registry
            .refresh_catalog()
            .await
            .expect("remove built-in override");
        let restored_event = updates.recv().await.expect("baseline restoration event");
        assert_eq!(restored_event.provider_ids, ["builtin"]);
        assert_eq!(
            catalog
                .provider_models("builtin")
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["builtin-base"]
        );
        assert_eq!(
            catalog
                .snapshot()
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["new-model", "builtin-base", "faux-runtime"]
        );
        let providers = registry.list_providers();
        assert_eq!(providers[0].name, "new-custom");
        assert_eq!(providers[1].name, "Built in");
        assert_eq!(providers[2].name, "faux");
    }

    #[test]
    fn test_provider_unregistration() {
        let registry = ModelRegistry::new();

        registry
            .register_builtin_provider("test".to_string(), "Test Provider".to_string(), vec![])
            .unwrap();
        assert!(registry.unregister_provider("test").is_ok());
        assert_eq!(registry.provider_count(), 0);
    }

    #[test]
    fn test_auth_status_update() {
        let registry = ModelRegistry::new();

        registry
            .register_builtin_provider("test".to_string(), "Test Provider".to_string(), vec![])
            .unwrap();

        assert!(registry
            .update_auth_status("test", AuthStatus::Authenticated)
            .is_ok());
        assert_eq!(
            registry.get_auth_status("test"),
            Some(AuthStatus::Authenticated)
        );
    }
}
