//! Built-in, package, and configured provider model catalog composition.

use std::collections::BTreeMap;

use rpi_ai::providers::anthropic::models::anthropic_models;
use rpi_ai::providers::openai_responses::openai_responses_models;
use rpi_ai::Model;
use rpi_extensions::ProviderManifest;

use crate::models::{provider_to_models, ConfigValueResolver, ModelsConfig, ProviderConfig};
/// Builds the model catalog from built-in providers, package manifests, and
/// host configuration while applying package defaults in place.
///
/// Credential expressions in model headers are resolved by the host and may
/// produce request-only secret material. Callers must remove headers before
/// exposing this catalog as public metadata.
///
/// # Parameters
/// - `config`: Mutable host-loaded provider configuration.
/// - `manifests`: Provider declarations from the package runtime.
/// - `resolver`: Host-owned configuration and environment resolver.
///
/// # Returns
/// The merged catalog, preserving built-in, package, and configured models.
pub fn build_catalog(
    config: &mut ModelsConfig,
    manifests: &[ProviderManifest],
    resolver: &dyn ConfigValueResolver,
) -> Vec<Model> {
    apply_provider_defaults(config, manifests);
    let mut catalog = anthropic_models();
    catalog.extend(openai_responses_models());
    for manifest in manifests {
        let provider_config = config.providers.get(&manifest.id);
        for mut model in manifest.models.clone() {
            if let Some(base_url) = provider_config.and_then(|provider| provider.base_url.as_ref())
            {
                model.base_url = base_url.clone();
            }
            if let Some(headers) = provider_config.and_then(|provider| provider.headers.as_ref()) {
                merge_model_headers(&mut model, resolve_headers(headers, resolver));
            }
            catalog.push(model);
        }
    }
    for (provider_id, provider_config) in &config.providers {
        let Some(models) = provider_to_models(provider_id, provider_config, resolver) else {
            continue;
        };
        for model in models {
            if let Some(existing) = catalog.iter_mut().find(|candidate| {
                candidate.api == model.api
                    && candidate.provider == model.provider
                    && candidate.id.eq_ignore_ascii_case(&model.id)
            }) {
                *existing = model;
            } else {
                catalog.push(model);
            }
        }
    }
    catalog
}

/// Resolves configured header values through the host's expression resolver.
///
/// # Parameters
/// - `headers`: Header expressions from provider configuration.
/// - `resolver`: Host-owned expression resolver.
///
/// # Returns
/// Resolved headers; unresolved expressions are omitted.
fn resolve_headers(
    headers: &BTreeMap<String, String>,
    resolver: &dyn ConfigValueResolver,
) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            resolver
                .resolve_uncached(value, None)
                .map(|resolved| (name.clone(), resolved))
        })
        .collect()
}

/// Merges configured authentication headers over model-declared headers.
///
/// Header names are matched case-insensitively so configured values replace
/// manifest defaults regardless of spelling.
///
/// # Parameters
/// - `model`: Model whose request headers are updated.
/// - `auth_headers`: Resolved provider-level headers.
fn merge_model_headers(model: &mut Model, auth_headers: BTreeMap<String, String>) {
    let headers = model.headers.get_or_insert_with(BTreeMap::new);
    for (name, value) in auth_headers {
        headers.retain(|existing, _| !existing.eq_ignore_ascii_case(&name));
        headers.insert(name, value);
    }
}

/// Applies package-declared provider defaults without replacing user values.
///
/// # Parameters
/// - `config`: Host-loaded mutable provider configuration.
/// - `manifests`: Provider declarations from the package runtime.
pub fn apply_provider_defaults(config: &mut ModelsConfig, manifests: &[ProviderManifest]) {
    for manifest in manifests {
        if !config.providers.contains_key(&manifest.id) {
            let first_model = manifest.models.first();
            let provider = ProviderConfig {
                name: None,
                base_url: first_model.map(|model| model.base_url.clone()),
                api_key: None,
                api: first_model.map(|model| model.api.as_str().to_string()),
                headers: None,
                auth_header: None,
                models: Vec::new(),
            };
            config.providers.insert(manifest.id.clone(), provider);
        }
        if let (Some(default), Some(provider)) = (
            manifest.models.first(),
            config.providers.get_mut(&manifest.id),
        ) {
            provider
                .api
                .get_or_insert_with(|| default.api.as_str().to_string());
            provider
                .base_url
                .get_or_insert_with(|| default.base_url.clone());
        }
    }
}

/// Resolves header expressions from one provider declaration.
///
/// # Parameters
/// - `headers`: Header expressions to resolve.
/// - `resolver`: Host-owned configuration resolver.
///
/// # Returns
/// The resolved header map, omitting unresolved values.
pub fn resolve_provider_headers(
    headers: &BTreeMap<String, String>,
    resolver: &dyn ConfigValueResolver,
) -> BTreeMap<String, String> {
    resolve_headers(headers, resolver)
}
