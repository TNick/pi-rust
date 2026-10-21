//! Host-injected model configuration refresh for the shared live catalog.

use rpi_ai::Model;
use rpi_extensions::ProviderManifest;

use crate::{ConfigValueResolver, ModelCatalog, ModelCatalogSnapshot, ModelsConfig, build_catalog};

impl ModelCatalog {
    /// Rebuilds and publishes models from caller-supplied provider configuration.
    ///
    /// The caller remains responsible for loading configuration, applying its
    /// credential policy, and deciding when to refresh. Provider defaults and
    /// model conversion use the shared upstream implementation. A host may
    /// prepare the resulting rows before publication; failure leaves both the
    /// live catalog and the caller's configuration unchanged.
    ///
    /// # Parameters
    /// - `config`: Caller-loaded provider configuration to update with defaults.
    /// - `manifests`: Provider declarations supplied by the caller.
    /// - `resolver`: Caller-owned endpoint, credential, and environment resolver.
    /// - `prepare`: Host policy applied before catalog publication.
    ///
    /// # Returns
    /// The exact coherent catalog snapshot committed by this refresh.
    ///
    /// # Errors
    /// Returns the host preparation error without publishing partial catalog
    /// state.
    pub fn refresh_from_configuration<E>(
        &self,
        config: &mut ModelsConfig,
        manifests: &[ProviderManifest],
        resolver: &dyn ConfigValueResolver,
        prepare: impl FnOnce(&mut Vec<Model>) -> Result<(), E>,
    ) -> Result<ModelCatalogSnapshot, E> {
        self.refresh_from_configuration_with_commit(
            config,
            manifests,
            resolver,
            prepare,
            |_| Ok(()),
        )
    }

    /// Prepares, commits host persistence, and publishes one configuration.
    ///
    /// The host persistence callback receives the fully defaulted candidate
    /// configuration after model preparation succeeds. If preparation or the
    /// callback fails, the live catalog and caller configuration are
    /// unchanged. The catalog is published only after the callback succeeds.
    ///
    /// # Parameters
    /// - `config`: Caller-loaded provider configuration to update with defaults.
    /// - `manifests`: Provider declarations supplied by the caller.
    /// - `resolver`: Caller-owned endpoint, credential, and environment resolver.
    /// - `prepare`: Host policy applied to candidate models before persistence.
    /// - `commit`: Host persistence or other fallible commit of candidate config.
    ///
    /// # Returns
    /// The exact coherent catalog snapshot committed by this refresh.
    ///
    /// # Errors
    /// Returns the host preparation or commit error without publishing partial
    /// catalog state.
    pub fn refresh_from_configuration_with_commit<E>(
        &self,
        config: &mut ModelsConfig,
        manifests: &[ProviderManifest],
        resolver: &dyn ConfigValueResolver,
        prepare: impl FnOnce(&mut Vec<Model>) -> Result<(), E>,
        commit: impl FnOnce(&ModelsConfig) -> Result<(), E>,
    ) -> Result<ModelCatalogSnapshot, E> {
        let mut next_config = config.clone();
        let mut models = build_catalog(&mut next_config, manifests, resolver);
        prepare(&mut models)?;
        commit(&next_config)?;
        let snapshot = self.reconcile_models_snapshot(models);
        *config = next_config;
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    //! Tests host-injected catalog refresh and failure atomicity.

    use std::collections::BTreeMap;

    use super::*;

    /// Resolves literal configuration without reading process state.
    struct LiteralResolver;

    impl ConfigValueResolver for LiteralResolver {
        /// Returns literal values unchanged.
        ///
        /// # Parameters
        /// - `value`: Configuration expression.
        /// - `environment`: Optional scoped environment.
        ///
        /// # Returns
        /// The unchanged literal value.
        fn resolve(
            &self,
            value: &str,
            _environment: Option<&BTreeMap<String, String>>,
        ) -> Option<String> {
            Some(value.to_owned())
        }

        /// Does not expose process environment to the refresh operation.
        ///
        /// # Parameters
        /// - `name`: Requested environment variable.
        ///
        /// # Returns
        /// No value because this resolver has no environment source.
        fn environment_value(&self, _name: &str) -> Option<String> {
            None
        }
    }

    /// Creates a caller-owned provider config for refresh coverage.
    fn configured_model() -> ModelsConfig {
        serde_json::from_value(serde_json::json!({
            "providers": {
                "sample": {
                    "api": "openai-completions",
                    "models": [{"id": "sample-model"}]
                }
            }
        }))
        .expect("valid provider config")
    }

    /// Publishes only host-prepared models using injected configuration inputs.
    #[test]
    fn refresh_publishes_prepared_configured_models() {
        let catalog = ModelCatalog::new();
        let mut config = configured_model();
        let snapshot = catalog
            .refresh_from_configuration(&mut config, &[], &LiteralResolver, |models| {
                models.retain(|model| model.provider == "sample");
                Ok::<(), ()>(())
            })
            .expect("refresh succeeds");

        assert_eq!(snapshot.revision, 1);
        assert_eq!(snapshot.models.len(), 1);
        assert_eq!(snapshot.models[0].provider, "sample");
        assert_eq!(snapshot.models[0].id, "sample-model");
    }

    /// Leaves config and live rows unchanged when host preparation rejects.
    #[test]
    fn failed_prepare_does_not_publish_or_apply_upstream_defaults() {
        let catalog = ModelCatalog::new();
        let mut config = configured_model();
        let original_config = serde_json::to_value(&config).expect("serializable config");
        let initial_snapshot = catalog.snapshot();
        let result = catalog.refresh_from_configuration(&mut config, &[], &LiteralResolver, |_| {
            Err("rejected by host policy")
        });

        assert_eq!(result, Err("rejected by host policy"));
        assert_eq!(
            serde_json::to_value(&config).expect("serializable config"),
            original_config
        );
        assert_eq!(catalog.snapshot(), initial_snapshot);
    }

    /// Leaves in-memory config and the published catalog unchanged when host
    /// persistence rejects a prepared candidate.
    #[test]
    fn failed_commit_does_not_publish_or_apply_upstream_defaults() {
        let catalog = ModelCatalog::new();
        let mut config = configured_model();
        let original_config = serde_json::to_value(&config).expect("serializable config");
        let initial_snapshot = catalog.snapshot();
        let result = catalog.refresh_from_configuration_with_commit(
            &mut config,
            &[],
            &LiteralResolver,
            |_| Ok::<(), &'static str>(()),
            |_| Err("persistence rejected candidate"),
        );

        assert_eq!(result, Err("persistence rejected candidate"));
        assert_eq!(
            serde_json::to_value(&config).expect("serializable config"),
            original_config
        );
        assert_eq!(catalog.snapshot(), initial_snapshot);
    }

    /// Passes the candidate to host persistence before publishing its models.
    #[test]
    fn successful_commit_publishes_prepared_configuration() {
        let catalog = ModelCatalog::new();
        let mut config = configured_model();
        let snapshot = catalog
            .refresh_from_configuration_with_commit(
                &mut config,
                &[],
                &LiteralResolver,
                |models| {
                    models.retain(|model| model.provider == "sample");
                    Ok::<(), &'static str>(())
                },
                |candidate| {
                    assert!(candidate.providers.contains_key("sample"));
                    Ok(())
                },
            )
            .expect("refresh and persistence succeed");

        assert_eq!(snapshot.revision, 1);
        assert_eq!(catalog.snapshot(), snapshot);
        assert!(config.providers.contains_key("sample"));
    }
}
