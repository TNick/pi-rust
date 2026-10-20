//! Provider and model configuration types plus conversion to `rpi-ai::Model`.

use std::collections::BTreeMap;
use std::fmt;

use rpi_ai::providers::anthropic::models::anthropic_models;
use rpi_ai::providers::openai_responses::openai_responses_models;
use rpi_ai::{Api, InputModality, Model, ModelCost, StreamingProtocolCompat, ThinkingLevelMap};
use rpi_extensions::ProviderManifest;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::credentials::{CredentialSource, ResolvedCredential};
use crate::{AuthStore, Credential};
/// Provider model configuration in the host's selected configuration source.
#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModelsConfig {
    /// Configured providers in declaration order.
    #[serde(default, skip_serializing_if = "indexmap::IndexMap::is_empty")]
    pub providers: indexmap::IndexMap<String, ProviderConfig>,
}

impl fmt::Debug for ModelsConfig {
    /// Formats provider identities while allowing each provider to redact data.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelsConfig")
            .field("providers", &self.providers)
            .finish()
    }
}

/// Configuration for one provider and its model definitions.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    /// Optional display name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Provider endpoint override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Provider credential expression, resolved by the host/runtime boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Provider protocol identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// Provider request headers, resolved by the host/runtime boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// Whether `api_key` is sent as a bearer authorization value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<bool>,
    /// Models declared for this provider.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ModelDefinition>,
}

impl fmt::Debug for ProviderConfig {
    /// Formats non-secret provider fields and redacts keys and headers.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("api", &self.api)
            .field("headers", &self.headers.as_ref().map(|_| "[redacted]"))
            .field("auth_header", &self.auth_header)
            .field("models", &self.models)
            .finish()
    }
}

/// Model metadata configured under one provider.
#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModelDefinition {
    /// Provider-native model identifier.
    pub id: String,
    /// Optional display name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Model endpoint override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Whether this model supports reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// Upstream thinking-level mapping for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    /// Context window size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Maximum output token count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Model token pricing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    /// Provider-specific sampling values understood by `rpi-ai::Model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<Map<String, Value>>,
    /// Supported input modality names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<String>>,
    /// Model request headers, resolved by the host/runtime boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// Provider-specific compatibility metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<Value>,
}

impl fmt::Debug for ModelDefinition {
    /// Formats model metadata and redacts request headers.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelDefinition")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("reasoning", &self.reasoning)
            .field("thinking_level_map", &self.thinking_level_map)
            .field("context_window", &self.context_window)
            .field("max_tokens", &self.max_tokens)
            .field("cost", &self.cost)
            .field("sampling_params", &self.sampling_params)
            .field("input", &self.input)
            .field("headers", &self.headers.as_ref().map(|_| "[redacted]"))
            .field("compat", &self.compat)
            .finish()
    }
}

/// Resolves host-specific configuration expressions without owning storage.
pub trait ConfigValueResolver: Send + Sync {
    /// Resolves one configuration expression with an optional scoped overlay.
    ///
    /// # Parameters
    /// - `value`: Literal or host-specific expression to resolve.
    /// - `environment`: Optional credential-scoped variable values.
    ///
    /// # Returns
    /// The resolved value, or `None` when a reference cannot be resolved.
    fn resolve(
        &self,
        value: &str,
        environment: Option<&BTreeMap<String, String>>,
    ) -> Option<String>;

    /// Resolves a value without reusing a cached host result.
    fn resolve_uncached(
        &self,
        value: &str,
        environment: Option<&BTreeMap<String, String>>,
    ) -> Option<String> {
        self.resolve(value, environment)
    }

    /// Reads one host environment variable without exposing the whole process
    /// environment to the runtime library.
    fn environment_value(&self, name: &str) -> Option<String>;
}

/// Resolves one provider's configured credential using upstream precedence.
///
/// Persistent credential storage and environment access remain host-injected.
/// The returned key is request-only data; callers must not place it in model
/// catalog snapshots or durable session records.
///
/// # Parameters
/// - `provider`: Stable provider identity.
/// - `config`: Optional provider-specific model configuration.
/// - `store`: Host-loaded credential store.
/// - `manifest`: Optional provider package declaration.
/// - `resolver`: Host configuration and environment resolver.
///
/// # Returns
/// The selected credential source and request secret, if available.
pub fn resolve_credential(
    provider: &str,
    config: Option<&ProviderConfig>,
    store: &AuthStore,
    manifest: Option<&ProviderManifest>,
    resolver: &dyn ConfigValueResolver,
) -> ResolvedCredential {
    let bearer = config
        .and_then(|provider_config| provider_config.auth_header)
        .unwrap_or(false);
    if let Some(credential) = store.get(provider) {
        match credential {
            Credential::ApiKey {
                key: Some(key),
                env,
            } => {
                if let Some(key) = nonempty(resolver.resolve(key, env.as_ref())) {
                    return resolved(key, CredentialSource::StoredApiKey, bearer);
                }
            }
            Credential::Oauth {
                access, expires, ..
            } if !access.trim().is_empty() && !credential_expired(*expires) => {
                return resolved(access.clone(), CredentialSource::StoredOauth, true);
            }
            _ => {}
        }
    }
    if let Some(key) = config
        .and_then(|provider_config| provider_config.api_key.as_deref())
        .and_then(|value| nonempty(resolver.resolve(value, None)))
    {
        return resolved(key, CredentialSource::ModelsJson, bearer);
    }
    if manifest.is_none() && provider == "anthropic" {
        if let Some(key) = nonempty(resolver.environment_value("ANTHROPIC_AUTH_TOKEN")) {
            return resolved(
                key,
                CredentialSource::Environment("ANTHROPIC_AUTH_TOKEN".into()),
                true,
            );
        }
    }
    let environment_name = match manifest {
        Some(provider_manifest) => provider_manifest.api_key_env.as_deref(),
        None => builtin_api_key_env(provider),
    };
    if let Some(name) = environment_name {
        if let Some(key) = nonempty(resolver.environment_value(name)) {
            return resolved(key, CredentialSource::Environment(name.into()), bearer);
        }
    }
    ResolvedCredential {
        key: None,
        source: if manifest.is_some_and(|provider_manifest| !provider_manifest.requires_auth) {
            CredentialSource::Anonymous
        } else {
            CredentialSource::Missing
        },
        bearer,
    }
}

/// Returns whether the provider identity has a built-in environment key.
pub fn builtin_api_key_env(provider: &str) -> Option<&'static str> {
    match provider {
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "openai" => Some("OPENAI_API_KEY"),
        _ => None,
    }
}

/// Checks whether a model already supplies a nonempty authentication header.
pub fn has_header_auth(headers: Option<&BTreeMap<String, String>>) -> bool {
    headers.is_some_and(|headers| {
        headers.iter().any(|(name, value)| {
            ["authorization", "x-api-key", "cf-aig-authorization"]
                .contains(&name.to_ascii_lowercase().as_str())
                && !value.trim().is_empty()
        })
    })
}

/// Returns whether a credential expiry timestamp is already past.
fn credential_expired(expires: i64) -> bool {
    if expires == 0 {
        return false;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(i64::MAX);
    expires <= now
}

/// Builds one credential result while keeping secret fields out of logs.
fn resolved(key: String, source: CredentialSource, bearer: bool) -> ResolvedCredential {
    ResolvedCredential {
        key: Some(key),
        source,
        bearer,
    }
}

/// Drops absent and whitespace-only credential values.
fn nonempty(value: Option<String>) -> Option<String> {
    value.filter(|key| !key.trim().is_empty())
}

/// Converts provider configuration into upstream model descriptors.
///
/// The host supplies `resolver` for environment or command-backed values. The
/// returned models contain resolved request headers and must not be used as a
/// public catalog snapshot without removing those headers.
///
/// # Parameters
/// - `provider_id`: Stable provider identity for these models.
/// - `config`: Provider configuration to convert.
/// - `resolver`: Host-supplied config expression resolver.
///
/// # Returns
/// The provider's model descriptors, or `None` for an unsupported protocol.
pub fn provider_to_models(
    provider_id: &str,
    config: &ProviderConfig,
    resolver: &dyn ConfigValueResolver,
) -> Option<Vec<Model>> {
    let api = match config.api.as_deref() {
        None | Some("") | Some("anthropic-messages") => Api::AnthropicMessages,
        Some("openai-completions") => Api::OpenaiCompletions,
        Some("openai-responses") => Api::OpenaiResponses,
        _ => return None,
    };
    let provider_base = config.base_url.clone().unwrap_or_else(|| match api {
        Api::OpenaiCompletions | Api::OpenaiResponses => "https://api.openai.com".into(),
        _ => "https://api.anthropic.com".into(),
    });
    let mut models = Vec::with_capacity(config.models.len());
    for definition in &config.models {
        let mut model = Model::new(
            definition.id.clone(),
            definition
                .name
                .clone()
                .unwrap_or_else(|| definition.id.clone()),
            api.clone(),
            provider_id,
            definition
                .base_url
                .clone()
                .unwrap_or_else(|| provider_base.clone()),
        );
        model.reasoning = definition.reasoning.unwrap_or(false);
        model.thinking_level_map = definition.thinking_level_map.clone();
        model.context_window = definition.context_window.unwrap_or(0);
        model.max_tokens = definition.max_tokens.unwrap_or(0);
        if let Some(cost) = &definition.cost {
            model.cost = cost.clone();
            model.cost_configured = true;
        }
        model.sampling_params = definition.sampling_params.clone();
        model.input = parse_input_modalities(definition.input.as_deref());
        let mut headers = BTreeMap::new();
        for source in [&definition.headers, &config.headers].into_iter().flatten() {
            for (name, value) in source {
                if let Some(value) = resolver.resolve_uncached(value, None) {
                    headers.insert(name.clone(), value);
                }
            }
        }
        if matches!(api, Api::OpenaiCompletions | Api::OpenaiResponses) {
            if let Some(value) = definition.compat.clone() {
                model.compat = if api == Api::OpenaiResponses {
                    serde_json::from_value(value)
                        .ok()
                        .map(StreamingProtocolCompat::OpenaiResponses)
                } else {
                    serde_json::from_value(value)
                        .ok()
                        .map(StreamingProtocolCompat::OpenaiCompletions)
                };
            }
        }
        if !headers.is_empty() {
            model.headers = Some(headers);
        }
        models.push(model);
    }
    Some(models)
}

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

/// Parses model input modality names, defaulting to text when none are known.
fn parse_input_modalities(input: Option<&[String]>) -> Vec<InputModality> {
    let modalities = input
        .unwrap_or(&[])
        .iter()
        .filter_map(|value| match value.to_ascii_lowercase().as_str() {
            "text" => Some(InputModality::Text),
            "image" => Some(InputModality::Image),
            _ => None,
        })
        .collect::<Vec<_>>();
    if modalities.is_empty() {
        vec![InputModality::Text]
    } else {
        modalities
    }
}
