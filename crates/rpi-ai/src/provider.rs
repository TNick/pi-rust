//! Mirrors `packages/ai/src/models.ts::Provider` — the trait every concrete
//! provider (faux, anthropic, …) implements. The agent loop never touches a
//! provider's wire format directly; it calls `stream_simple` and consumes the
//! returned [`AssistantMessageEventStream`].

use crate::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use crate::model::Model;
use crate::provider_option_attestation::ProviderOptionsAttestation;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, Context, DeferredHandle, ErrorReason, StopReason,
    ThinkingBudgets, ThinkingLevel,
};
use sha2::{Digest, Sha256};
use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use zeroize::Zeroize;

/// Provider-owned opaque state for continuing a prior response without replaying
/// the full conversation context.
///
/// The handle is scoped to one provider and model. Its private state is
/// in-memory only: it has no `Serialize` implementation and its `Debug`
/// representation redacts the state. Providers should store only a minimal
/// opaque continuation reference here. Do not store request/response wire
/// payloads, credentials, or secrets in the state. The host owns lineage
/// validation: it must only reuse a handle when the current context belongs to
/// the same logical conversation or branch that produced it.
#[derive(Clone)]
pub struct ProviderContinuation {
    provider_id: String,
    model_id: String,
    state: Arc<dyn Any + Send + Sync>,
}

impl ProviderContinuation {
    /// Creates a provider/model-scoped handle around provider-owned state.
    pub fn new<T>(provider_id: impl Into<String>, model_id: impl Into<String>, state: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            provider_id: provider_id.into(),
            model_id: model_id.into(),
            state: Arc::new(state),
        }
    }

    /// Returns the provider identity that created this handle.
    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    /// Returns the model identity that created this handle.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns whether this handle belongs to the requested provider and model.
    pub fn is_scoped_to(&self, provider_id: &str, model_id: &str) -> bool {
        self.provider_id == provider_id && self.model_id == model_id
    }

    /// Borrows the provider's concrete state for request construction.
    ///
    /// The callback is the only access to the opaque state. Providers should
    /// use a private state type and extract only the continuation reference
    /// required by their own protocol.
    pub fn with_state<T, R>(&self, use_state: impl FnOnce(&T) -> R) -> Option<R>
    where
        T: Any + Send + Sync,
    {
        let state = Arc::clone(&self.state).downcast::<T>().ok()?;
        Some(use_state(&state))
    }
}

impl fmt::Debug for ProviderContinuation {
    /// Formats handle scope while keeping provider-owned state opaque.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderContinuation")
            .field("provider_id", &self.provider_id)
            .field("model_id", &self.model_id)
            .field("state", &"[opaque]")
            .finish()
    }
}

impl PartialEq for ProviderContinuation {
    /// Clones compare equal while independently created states remain distinct.
    fn eq(&self, other: &Self) -> bool {
        self.provider_id == other.provider_id
            && self.model_id == other.model_id
            && Arc::ptr_eq(&self.state, &other.state)
    }
}

impl Eq for ProviderContinuation {}

/// Rejects opaque options unless a provider defines explicit handling.
///
/// # Parameters
/// - `model`: Model whose opaque provider options may be unsupported.
/// - `provider_id`: Identity of the provider receiving the request.
///
/// # Returns
/// An error stream when options are present, otherwise `None`.
pub(crate) fn reject_opaque_provider_options(
    model: &Model,
    provider_id: &str,
) -> Option<AssistantMessageEventStream> {
    if model.provider_options.is_empty() {
        return None;
    }
    let (mut producer, stream) = create_assistant_message_event_stream();
    producer.push(AssistantMessageEvent::Error {
        reason: ErrorReason::Error,
        error: AssistantMessage::terminal(
            model.api.clone(),
            provider_id,
            model.id.clone(),
            StopReason::Error,
            "provider does not support opaque model options",
            0,
        ),
    });
    Some(stream)
}

/// Publishes safe settings and opaque digests of the final provider request.
///
/// Values are read from the body after provider transformation. Prompt and
/// content fields, arbitrary header values, credentials, arbitrary sampling
/// values, and provider-specific opaque fields are never copied. The body
/// contributes only a digest; a narrow set of fixed, non-sensitive header
/// values contributes a separate digest. The report remains incomplete because
/// endpoint and unclassified header identity are omitted.
///
/// # Parameters
/// - `model`: Model descriptor used to prepare the request.
/// - `callback`: Optional sink for the report.
/// - `body`: Final JSON request body immediately before transport.
/// - `headers`: Final headers immediately before protected credentials are
///   inserted by the transport bridge.
pub(crate) fn report_applied_request_settings<'a, I>(
    model: &Model,
    callback: Option<&AppliedRequestSettingsCallback>,
    body: &serde_json::Value,
    headers: I,
) where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let Some(callback) = callback else {
        return;
    };
    let mut applied_values = BTreeMap::new();
    for key in [
        "model",
        "previous_response_id",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "reasoning",
        "thinking",
        "stream",
        "store",
    ] {
        if let Some(value) = body.get(key) {
            let valid_scalar = match key {
                "model" | "previous_response_id" => value.is_string(),
                "temperature" | "top_p" | "frequency_penalty" | "presence_penalty" => {
                    value.is_number()
                }
                "seed" => value.as_i64().is_some() || value.as_u64().is_some(),
                "max_tokens" | "max_completion_tokens" | "max_output_tokens" => {
                    value.as_u64().is_some()
                }
                "stream" | "store" => value.is_boolean(),
                _ => false,
            };
            if valid_scalar {
                applied_values.insert(key.to_owned(), value.clone());
            } else if key == "reasoning" {
                if let Some(effort) = value.get("effort").filter(|value| value.is_string()) {
                    applied_values.insert(key.to_owned(), serde_json::json!({ "effort": effort }));
                }
            } else if key == "thinking" {
                let mut projection = serde_json::Map::new();
                if let Some(kind) = value.get("type").filter(|value| value.is_string()) {
                    projection.insert("type".to_owned(), kind.clone());
                }
                if let Some(budget) = value
                    .get("budget_tokens")
                    .filter(|value| value.as_u64().is_some())
                {
                    projection.insert("budget_tokens".to_owned(), budget.clone());
                }
                if !projection.is_empty() {
                    applied_values.insert(key.to_owned(), projection.into());
                }
            }
        }
    }
    let request_body_sha256 = serde_json::to_vec(body).ok().map(|bytes| {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    });
    let mut safe_headers = BTreeMap::new();
    let mut unclassified_headers = false;
    for (name, value) in headers {
        let key = name.to_ascii_lowercase();
        let safe_value = match key.as_str() {
            "accept" if matches!(value, "application/json" | "text/event-stream") => Some(value),
            "content-type" if value == "application/json" => Some(value),
            "anthropic-version" if value == "2023-06-01" => Some(value),
            "authorization" | "proxy-authorization" | "cookie" | "set-cookie" | "x-api-key" => None,
            _ => {
                unclassified_headers = true;
                None
            }
        };
        if let Some(safe_value) = safe_value {
            safe_headers.insert(key, safe_value.to_owned());
        }
    }
    let request_header_sha256 = serde_json::to_vec(&safe_headers).ok().map(|bytes| {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    });
    let mut omitted_categories = vec![
        "endpoint".to_owned(),
        "headers_and_credentials".to_owned(),
        "prompt_and_transcript".to_owned(),
        "tools_and_schemas".to_owned(),
        "opaque_provider_options".to_owned(),
        "unrecognized_request_fields".to_owned(),
    ];
    if unclassified_headers {
        omitted_categories.push("unclassified_headers".to_owned());
    }
    callback(AppliedRequestSettings {
        schema_version: "rpi-applied-request-settings-v4".to_owned(),
        provider_id: model.provider.clone(),
        model_id: model.id.clone(),
        api_id: model.api.as_str().to_owned(),
        applied_values,
        request_body_sha256,
        request_header_sha256,
        request_effect_sha256: None,
        provider_options: ProviderOptionsAttestation::default(),
        omitted_categories,
        complete: false,
    });
}

/// Reports a version-five prepared-request identity when safe coverage allows it.
///
/// The digest covers the final URL, JSON body, and provider-controlled headers.
/// Credential material keys the digest and is never copied into the report.
///
/// # Parameters
/// - `model`: Model descriptor used to prepare the request.
/// - `callback`: Optional sink for the sanitized report.
/// - `url`: Fully resolved provider request URL.
/// - `body`: Final JSON request body.
/// - `headers`: Final provider headers before transport-added headers.
/// - `request_secret`: Optional host-protected credential and auth mode.
pub(crate) fn report_applied_request_settings_with_effect<'a, I>(
    model: &Model,
    callback: Option<&AppliedRequestSettingsCallback>,
    url: &str,
    body: &serde_json::Value,
    headers: I,
    request_secret: Option<(&RequestSecret, bool)>,
) where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let Some(callback) = callback else {
        return;
    };
    let mut headers = headers
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect::<BTreeMap<_, _>>();
    let report_slot = Arc::new(std::sync::Mutex::new(None));
    let report_sink = Arc::clone(&report_slot);
    let capture: AppliedRequestSettingsCallback = Arc::new(move |report| {
        *report_sink
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(report);
    });
    report_applied_request_settings(
        model,
        Some(&capture),
        body,
        headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str())),
    );
    let Some(mut report) = report_slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    else {
        return;
    };
    report.schema_version = "rpi-applied-request-settings-v5".to_owned();
    report.request_effect_sha256 =
        crate::request_effect_identity::fingerprint(url, body, &headers, request_secret);
    if report.request_effect_sha256.is_some() {
        report.omitted_categories.clear();
        report.complete = true;
    } else {
        report
            .omitted_categories
            .push("request_effect_identity".to_owned());
        report.complete = false;
    }
    callback(report);
    for value in headers.values_mut() {
        value.zeroize();
    }
}

/// Default deadline for one LLM API request, including its streamed response.
pub const DEFAULT_LLM_API_TIMEOUT: Duration = Duration::from_secs(600);

/// Reports sanitized settings observed at a provider's final request boundary.
///
/// # Public members
/// - `schema_version`: Version of this report contract.
/// - `provider_id`: Provider identity used for the request.
/// - `model_id`: Model identifier placed in the request.
/// - `api_id`: Provider API identifier.
/// - `applied_values`: Allowlisted scalar settings found in the final body.
/// - `request_body_sha256`: Digest of the exact prepared JSON body.
/// - `request_header_sha256`: Digest of allowlisted prepared header values.
/// - `provider_options`: Opaque per-option application status and value digests.
/// - `omitted_categories`: Sensitive or opaque request categories omitted.
/// - `complete`: Whether every request-affecting setting is represented.
///
/// # Private members
/// None.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppliedRequestSettings {
    /// Version of this report contract.
    pub schema_version: String,
    /// Provider identity used by the request.
    pub provider_id: String,
    /// Model identifier used by the request.
    pub model_id: String,
    /// Provider API identifier.
    pub api_id: String,
    /// Sanitized allowlisted settings observed in the final request body.
    pub applied_values: BTreeMap<String, serde_json::Value>,
    /// SHA-256 identity of the exact prepared JSON body, without storing it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_body_sha256: Option<String>,
    /// SHA-256 identity of exact allowlisted, non-sensitive header values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_header_sha256: Option<String>,
    /// Opaque identity of the exact prepared request, including credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_effect_sha256: Option<String>,
    /// Opaque applied/transformed/ignored status for schema-safe provider options.
    #[serde(default)]
    pub provider_options: ProviderOptionsAttestation,
    /// Omitted request categories that may contain secrets or opaque data.
    pub omitted_categories: Vec<String>,
    /// Whether the report attests every request-affecting value.
    pub complete: bool,
}

/// Callback invoked after a provider assembles the final JSON request body.
pub type AppliedRequestSettingsCallback = Arc<dyn Fn(AppliedRequestSettings) + Send + Sync>;

/// Cache-retention hint passed to a provider. Mirrors TS `CacheRetention`.
/// `Long` requests 1h ephemeral cache control on Anthropic (subject to compat);
/// `Short` requests the default ephemeral; `None` disables cache control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheRetention {
    None,
    #[default]
    Short,
    Long,
}

/// Options for a "simple" (non-agent) stream call — what compaction / smoke
/// tests / bare provider usage pass. The agent loop wraps this with its richer
/// equivalent when it drives a real turn (M2).
#[derive(Clone)]
pub struct SimpleStreamOptions {
    /// API key. A provider may fall back to an env var when `None` (M3 anthropic
    /// reads `ANTHROPIC_API_KEY`); faux ignores it.
    pub api_key: Option<String>,
    /// Request timeout. Defaults to 600 seconds; `None` uses the same provider
    /// fallback and can be used by patches to clear a caller override.
    pub timeout: Option<Duration>,
    /// Max retry attempts on retryable errors (408/409/429/>=500). `None` = 0.
    pub max_retries: Option<u32>,
    /// Cap on the total backoff between retries. `None` = 60s.
    pub max_retry_delay: Option<Duration>,
    /// Extra HTTP headers to send (merged over provider defaults).
    pub headers: Option<BTreeMap<String, String>>,
    /// Free-form provider metadata (request id, trace id, …). Anthropic reads
    /// `user_id` for abuse tracking; faux ignores it.
    pub metadata: Option<BTreeMap<String, String>>,
    /// Cache-retention hint for prompt caching.
    pub cache_retention: CacheRetention,
    /// Optional session id (faux uses it to simulate prompt-cache hits).
    pub session_id: Option<String>,
    /// Cancellation token. Providers must poll this between chunks and emit an
    /// `Error { reason: Aborted }` terminal event when cancelled. Defaults to a
    /// fresh un-cancelled token so `Default`-constructed options never abort.
    pub signal: CancellationToken,
    /// Extended-thinking level. `None`/`Off` → thinking disabled (provider may
    /// still emit `thinking: { type: "disabled" }` when the model reasons).
    /// Mirrors TS `SimpleStreamOptions.reasoning`.
    pub reasoning: Option<ThinkingLevel>,
    /// Optional caller override for the requested output-token ceiling. `None`
    /// lets the provider fit the thinking budget inside `Model::max_tokens`.
    /// Mirrors TS `SimpleStreamOptions.maxTokens`.
    pub max_tokens: Option<u64>,
    /// Sampling temperature. Only applied when the model supports it AND
    /// thinking is disabled (Anthropic rejects temperature under thinking).
    /// Mirrors TS `StreamOptions.temperature`.
    pub temperature: Option<f64>,
    /// Per-level reasoning token budgets for budget-based thinking models.
    /// Faux ignores this; anthropic uses it in `adjust_max_tokens_for_thinking`.
    pub thinking_budgets: Option<ThinkingBudgets>,
}

impl fmt::Debug for SimpleStreamOptions {
    /// Formats safe request options while redacting keys and header values.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header_names = self
            .headers
            .as_ref()
            .map(|headers| headers.keys().map(String::as_str).collect::<Vec<_>>());
        let metadata_names = self
            .metadata
            .as_ref()
            .map(|metadata| metadata.keys().map(String::as_str).collect::<Vec<_>>());
        formatter
            .debug_struct("SimpleStreamOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("max_retry_delay", &self.max_retry_delay)
            .field("header_names", &header_names)
            .field("metadata_names", &metadata_names)
            .field("cache_retention", &self.cache_retention)
            .field("session_id", &self.session_id.as_ref().map(|_| "[present]"))
            .field("signal", &"[cancellation token]")
            .field("reasoning", &self.reasoning)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("thinking_budgets", &self.thinking_budgets)
            .finish()
    }
}

/// Host-protected secret material that can be exposed only to request code.
///
/// Implementations should expose the value only for the duration of the
/// callback and must not include it in formatting or serialization.
pub trait SecretMaterial: Send + Sync {
    /// Makes the secret available to one request-building callback.
    ///
    /// # Parameters
    /// - `consumer`: Request code that consumes the secret synchronously.
    fn with_secret(&self, consumer: &mut dyn FnMut(&str));
}

/// Opaque request-only provider secret supplied by a host.
pub struct RequestSecret(std::sync::Arc<dyn SecretMaterial>);

impl RequestSecret {
    /// Wraps host-protected material for a provider request.
    ///
    /// # Parameters
    /// - `material`: Host-owned secret material implementation.
    ///
    /// # Returns
    /// A cloneable request-only secret handle.
    pub fn new(material: std::sync::Arc<dyn SecretMaterial>) -> Self {
        Self(material)
    }

    /// Lets provider request code consume the secret synchronously.
    ///
    /// # Parameters
    /// - `consumer`: Request code that consumes the secret.
    pub fn with_secret(&self, consumer: &mut dyn FnMut(&str)) {
        self.0.with_secret(consumer);
    }
}

impl Clone for RequestSecret {
    /// Clones the opaque reference without copying secret bytes.
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl fmt::Debug for RequestSecret {
    /// Formats the request secret without revealing its value.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RequestSecret([redacted])")
    }
}

impl Default for SimpleStreamOptions {
    fn default() -> Self {
        Self {
            api_key: None,
            timeout: Some(DEFAULT_LLM_API_TIMEOUT),
            max_retries: None,
            max_retry_delay: None,
            headers: None,
            metadata: None,
            cache_retention: CacheRetention::default(),
            session_id: None,
            signal: CancellationToken::new(),
            reasoning: None,
            max_tokens: None,
            temperature: None,
            thinking_budgets: None,
        }
    }
}

impl SimpleStreamOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_signal(mut self, signal: CancellationToken) -> Self {
        self.signal = signal;
        self
    }

    /// Resolve the effective reasoning level, defaulting to `Off` when unset.
    pub fn reasoning_level(&self) -> ThinkingLevel {
        self.reasoning.unwrap_or(ThinkingLevel::Off)
    }

    /// Resolve the effective request timeout, including the provider fallback.
    pub fn request_timeout(&self) -> Duration {
        self.timeout.unwrap_or(DEFAULT_LLM_API_TIMEOUT)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        report_applied_request_settings, report_applied_request_settings_with_effect,
        AppliedRequestSettings, AppliedRequestSettingsCallback, Provider, ProviderContinuation,
        RequestSecret, SecretMaterial, SimpleStreamOptions, DEFAULT_LLM_API_TIMEOUT,
    };
    use crate::model::Model;
    use crate::types::Api;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Supplies one credential to the request-report test.
    struct TestSecret(String);

    impl SecretMaterial for TestSecret {
        /// Exposes the test credential during the request callback.
        ///
        /// # Parameters
        /// - `consumer`: Callback that reads the credential synchronously.
        fn with_secret(&self, consumer: &mut dyn FnMut(&str)) {
            consumer(&self.0);
        }
    }

    /// Verifies complete reports carry only an opaque effect identity.
    #[test]
    fn complete_report_contains_keyed_request_effect_without_credentials() {
        let model = Model::new(
            "model-a",
            "Model A",
            Api::OpenaiCompletions,
            "provider-a",
            "https://example.invalid/v1",
        );
        let reports = Arc::new(Mutex::new(Vec::<AppliedRequestSettings>::new()));
        let sink = Arc::clone(&reports);
        let callback: AppliedRequestSettingsCallback =
            Arc::new(move |report| sink.lock().unwrap().push(report));
        let request_secret = RequestSecret::new(Arc::new(TestSecret(
            "credential-report-sentinel".to_owned(),
        )));
        report_applied_request_settings_with_effect(
            &model,
            Some(&callback),
            "https://example.invalid/v1/chat/completions",
            &json!({"model": "model-a", "input": "private prompt sentinel"}),
            [
                ("content-type", "application/json"),
                ("authorization", "Bearer credential-report-sentinel"),
            ],
            Some((&request_secret, true)),
        );

        let reports = reports.lock().unwrap();
        let report = &reports[0];
        assert_eq!(report.schema_version, "rpi-applied-request-settings-v5");
        assert!(report.complete);
        assert!(report.omitted_categories.is_empty());
        assert_eq!(report.request_effect_sha256.as_deref().unwrap().len(), 64);
        let encoded = serde_json::to_string(report).expect("report JSON");
        assert!(!encoded.contains("credential-report-sentinel"));
        assert!(!encoded.contains("private prompt sentinel"));
    }

    /// Captures exact final-body identity without retaining request content.
    #[test]
    fn applied_settings_hash_the_prepared_body_without_serializing_it() {
        let model = Model::new(
            "model-a",
            "Model A",
            Api::OpenaiCompletions,
            "provider-a",
            "https://example.invalid/v1",
        );
        let reports = Arc::new(Mutex::new(Vec::<AppliedRequestSettings>::new()));
        let sink = Arc::clone(&reports);
        let callback: AppliedRequestSettingsCallback =
            Arc::new(move |report| sink.lock().unwrap().push(report));
        let first_body = json!({
            "model": "model-a",
            "temperature": 0.5,
            "input": "private prompt sentinel"
        });
        let second_body = json!({
            "model": "model-a",
            "temperature": 0.5,
            "input": "different private prompt"
        });

        report_applied_request_settings(
            &model,
            Some(&callback),
            &first_body,
            [("content-type", "application/json")],
        );
        report_applied_request_settings(
            &model,
            Some(&callback),
            &second_body,
            [("accept", "text/event-stream")],
        );

        let reports = reports.lock().unwrap();
        assert_eq!(reports[0].schema_version, "rpi-applied-request-settings-v4");
        let first_digest = reports[0]
            .request_body_sha256
            .as_deref()
            .expect("prepared body digest");
        let second_digest = reports[1]
            .request_body_sha256
            .as_deref()
            .expect("prepared body digest");
        assert_eq!(first_digest.len(), 64);
        assert_ne!(first_digest, second_digest);
        let first_headers = reports[0]
            .request_header_sha256
            .as_deref()
            .expect("allowlisted header digest");
        let second_headers = reports[1]
            .request_header_sha256
            .as_deref()
            .expect("allowlisted header digest");
        assert_ne!(first_headers, second_headers);
        let encoded = serde_json::to_string(&reports[0]).expect("report JSON");
        assert!(!encoded.contains("private prompt sentinel"));
        assert!(!reports[0].complete);
    }

    #[test]
    fn prepared_header_digest_omits_credentials_and_unclassified_values() {
        let model = Model::new(
            "model-a",
            "Model A",
            Api::OpenaiCompletions,
            "provider-a",
            "https://example.invalid/v1",
        );
        let reports = Arc::new(Mutex::new(Vec::<AppliedRequestSettings>::new()));
        let sink = Arc::clone(&reports);
        let callback: AppliedRequestSettingsCallback =
            Arc::new(move |report| sink.lock().unwrap().push(report));
        let body = json!({"model": "model-a"});
        report_applied_request_settings(
            &model,
            Some(&callback),
            &body,
            [
                ("content-type", "application/json"),
                ("authorization", "Bearer private-header-sentinel"),
                ("x-custom", "private-custom-sentinel"),
            ],
        );

        let reports = reports.lock().unwrap();
        let report = &reports[0];
        let encoded = serde_json::to_string(report).expect("report JSON");
        assert!(!encoded.contains("private-header-sentinel"));
        assert!(!encoded.contains("private-custom-sentinel"));
        assert!(report
            .omitted_categories
            .contains(&"unclassified_headers".to_owned()));
        assert!(!report.complete);
    }

    #[test]
    fn llm_api_timeout_defaults_to_ten_minutes() {
        let options = SimpleStreamOptions::default();
        assert_eq!(options.timeout, Some(Duration::from_secs(600)));
        assert_eq!(options.request_timeout(), DEFAULT_LLM_API_TIMEOUT);
    }

    #[test]
    fn llm_api_timeout_allows_override_and_none_uses_provider_fallback() {
        let mut options = SimpleStreamOptions::default();
        options.timeout = Some(Duration::from_secs(15));
        assert_eq!(options.request_timeout(), Duration::from_secs(15));

        options.timeout = None;
        assert_eq!(options.request_timeout(), Duration::from_secs(600));
    }

    #[test]
    fn provider_continuation_is_scoped_and_redacts_opaque_state() {
        let continuation = ProviderContinuation::new(
            "provider-a",
            "model-a",
            "private-continuation-reference".to_owned(),
        );
        let clone = continuation.clone();

        assert!(continuation.is_scoped_to("provider-a", "model-a"));
        assert!(!continuation.is_scoped_to("provider-b", "model-a"));
        assert!(!continuation.is_scoped_to("provider-a", "model-b"));
        assert_eq!(continuation, clone);
        assert_eq!(
            continuation.with_state::<String, _>(String::clone),
            Some("private-continuation-reference".to_owned())
        );
        let debug = format!("{continuation:?}");
        assert!(debug.contains("provider-a"));
        assert!(!debug.contains("private-continuation-reference"));
    }

    #[tokio::test]
    async fn default_provider_continuation_rejects_out_of_scope_handles() {
        use crate::providers::faux::{FauxProvider, FauxScript};
        use crate::types::AssistantMessageEvent;

        let provider = FauxProvider::new(FauxScript::new());
        let model = provider.default_model().clone();
        let continuation =
            ProviderContinuation::new("another-provider", model.id.clone(), "opaque-state");
        let mut stream = provider
            .stream_simple_with_continuation(&model, &continuation, &SimpleStreamOptions::default())
            .await;

        let event = stream.next().await.expect("continuation error event");
        let AssistantMessageEvent::Error { error, .. } = event else {
            panic!("out-of-scope continuation should fail closed");
        };
        assert_eq!(
            error.error_message.as_deref(),
            Some("provider continuation scope does not match this model")
        );
    }
}

/// The trait seam. The arrow `pi-agent → pi-ai` means the agent loop depends on
/// this trait alone — provider crates are pluggable behind it.
///
/// `stream_simple` returns synchronously (no `await`): it creates an
/// [`AssistantMessageEventStream`], spawns the producer task internally, and
/// hands back the consumer. Failures surface as `Error` events on the stream,
/// never as panics or `Err` from this method.
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Stable provider id (e.g. `"anthropic"`, `"faux"`).
    fn id(&self) -> &str;

    /// Models this provider serves.
    fn models(&self) -> &[Model];

    /// Reports whether audited response continuation is supported.
    ///
    /// Hosts must check this capability before calling
    /// `stream_simple_with_continuation_and_applied_settings` with `None`,
    /// since that call opts into provider-side response retention. The default
    /// is `false`; providers opt in only when they implement the audited
    /// continuation path.
    ///
    /// # Returns
    /// `true` when audited continuation requests are supported.
    fn supports_response_continuation(&self) -> bool {
        false
    }

    /// Stream a single assistant turn against `ctx` on `model`. Returns the
    /// consumer end of an event stream; the producer task is already running.
    async fn stream_simple(
        &self,
        model: &Model,
        ctx: &Context,
        opts: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream;

    /// Extracts provider-owned continuation state from a completed response.
    ///
    /// Implementations may return a handle only when the response can be
    /// continued without retaining its wire payload or credentials. The
    /// returned handle is in-memory only and must remain scoped to this
    /// provider and the supplied model. Overrides of
    /// `stream_simple_with_continuation` must verify this scope before reading
    /// provider-owned state.
    fn continuation_from_response(
        &self,
        _model: &Model,
        _response: &AssistantMessage,
    ) -> Option<ProviderContinuation> {
        None
    }

    /// Continues a provider response without a conversation context.
    ///
    /// This is a separate capability from [`DeferredProvider`], which polls a
    /// response that has not completed yet. The default fails closed so a
    /// provider cannot accidentally ignore a continuation or replay an
    /// unrelated conversation.
    ///
    /// # Parameters
    /// - `model`: Model that must match the handle's scope.
    /// - `continuation`: Opaque provider-owned state previously returned by
    ///   [`Provider::continuation_from_response`].
    /// - `opts`: Request options. Credentials are supplied separately by the
    ///   host/provider and are never stored in the continuation handle.
    ///
    /// # Returns
    /// The continued response event stream, or a safe error stream if the
    /// handle is out of scope or the provider does not implement continuation.
    async fn stream_simple_with_continuation(
        &self,
        model: &Model,
        continuation: &ProviderContinuation,
        _opts: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let message = if continuation.is_scoped_to(self.id(), &model.id) {
            "provider does not support response continuation"
        } else {
            "provider continuation scope does not match this model"
        };
        continuation_error_stream(model, self.id(), message)
    }

    /// Streams with opt-in response continuation and an audited request boundary.
    ///
    /// Providers must use `ctx` to identify new input while honoring the
    /// optional opaque continuation, consume protected credentials only during
    /// request construction, and report final request settings through
    /// `callback`. Calling this method with `None` opts in to any provider-side
    /// response retention required for a later continuation. The default fails
    /// closed so continuation cannot bypass auth or reporting by falling back
    /// to the legacy continuation method. Hosts are responsible for validating
    /// that a supplied handle's conversation lineage matches `ctx`; provider
    /// and model scope alone cannot establish that relationship.
    ///
    /// # Parameters
    /// - `model`: Model that must match the continuation scope.
    /// - `ctx`: Current provider-neutral context for the next request.
    /// - `continuation`: Optional opaque handle produced by this provider.
    /// - `opts`: Non-secret request options.
    /// - `request_secret`: Optional host-protected credential and bearer mode.
    /// - `callback`: Required sink for sanitized final request settings.
    ///
    /// # Returns
    /// The continued response event stream or a safe unsupported error.
    async fn stream_simple_with_continuation_and_applied_settings(
        &self,
        model: &Model,
        _ctx: &Context,
        continuation: Option<&ProviderContinuation>,
        _opts: &SimpleStreamOptions,
        _request_secret: Option<(&RequestSecret, bool)>,
        _callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        let message = match continuation {
            Some(continuation) if !continuation.is_scoped_to(self.id(), &model.id) => {
                "provider continuation scope does not match this model"
            }
            _ => "provider does not support audited response continuation",
        };
        continuation_error_stream(model, self.id(), message)
    }

    /// Stream a request and report sanitized settings from its final body.
    ///
    /// Providers that do not expose request preparation keep their existing
    /// behavior through this default. Built-in HTTP providers override it and
    /// invoke the callback only after their final body transformations.
    ///
    /// # Parameters
    /// - `model`: Model to execute.
    /// - `ctx`: Prompt context.
    /// - `opts`: Non-secret request options.
    /// - `callback`: Sink for the versioned applied-settings report.
    ///
    /// # Returns
    /// The provider response event stream.
    async fn stream_simple_with_applied_settings(
        &self,
        model: &Model,
        ctx: &Context,
        opts: &SimpleStreamOptions,
        _callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        self.stream_simple(model, ctx, opts).await
    }

    /// Streams with opaque host-protected request credentials when supported.
    ///
    /// Implementations must consume the secret only while constructing the
    /// provider request and must not copy it into model metadata, generic
    /// options, logs, or serialized extension envelopes. The default reports
    /// unsupported so credentials never fall back to cloneable string options.
    ///
    /// # Parameters
    /// - `model`: Model to execute.
    /// - `ctx`: Prompt context.
    /// - `opts`: Non-secret request options.
    /// - `secret`: Opaque host-protected credential.
    /// - `bearer`: Whether the provider should send bearer authorization.
    ///
    /// # Returns
    /// The provider response event stream, or a safe unsupported error.
    async fn stream_simple_with_request_secret(
        &self,
        model: &Model,
        _ctx: &Context,
        _opts: &SimpleStreamOptions,
        _secret: &RequestSecret,
        _bearer: bool,
    ) -> AssistantMessageEventStream {
        let (mut producer, stream) = create_assistant_message_event_stream();
        let error = AssistantMessage::terminal(
            model.api.clone(),
            self.id(),
            model.id.clone(),
            StopReason::Error,
            "provider does not support protected credential input",
            0,
        );
        producer.push(AssistantMessageEvent::Error {
            reason: ErrorReason::Error,
            error,
        });
        stream
    }

    /// Stream a protected-credential request and report its final body settings.
    ///
    /// # Parameters
    /// - `model`: Model to execute.
    /// - `ctx`: Prompt context.
    /// - `opts`: Non-secret request options.
    /// - `secret`: Opaque host-protected credential.
    /// - `bearer`: Whether the credential uses bearer authorization.
    /// - `callback`: Sink for the versioned applied-settings report.
    ///
    /// # Returns
    /// The provider response event stream.
    async fn stream_simple_with_request_secret_and_applied_settings(
        &self,
        model: &Model,
        ctx: &Context,
        opts: &SimpleStreamOptions,
        secret: &RequestSecret,
        bearer: bool,
        _callback: AppliedRequestSettingsCallback,
    ) -> AssistantMessageEventStream {
        self.stream_simple_with_request_secret(model, ctx, opts, secret, bearer)
            .await
    }

    /// This provider's deferred-response capability, if it has one.
    ///
    /// Defaulted rather than required, and returned as an optional sidecar:
    /// only providers whose API actually has a long-poll continuation can
    /// answer, and the rest should not carry a stub. This is the same reasoning
    /// that put [`ProviderHooks`] in its own trait, except a default body also
    /// spares every existing implementation.
    fn deferred(&self) -> Option<&dyn DeferredProvider> {
        None
    }
}

/// Creates a terminal error stream for a rejected continuation request.
pub(crate) fn continuation_error_stream(
    model: &Model,
    provider_id: &str,
    message: &str,
) -> AssistantMessageEventStream {
    let (mut producer, stream) = create_assistant_message_event_stream();
    producer.push(AssistantMessageEvent::Error {
        reason: ErrorReason::Error,
        error: AssistantMessage::terminal(
            model.api.clone(),
            provider_id,
            model.id.clone(),
            StopReason::Error,
            message,
            0,
        ),
    });
    stream
}

/// Starts a provider stream with shared cancellation and terminal-event repair.
///
/// # Parameters
/// - `provider`: Provider selected by the host.
/// - `model`: Exact provider model authorized by the host.
/// - `context`: Provider-neutral request context.
/// - `options`: Non-secret stream options.
/// - `request_secret`: Optional protected credential and bearer mode.
/// - `applied_settings`: Optional sanitized applied-settings callback.
/// - `continuation`: Optional provider/model-scoped continuation handle.
/// - `continuation_enabled`: Whether the host opted into audited continuation.
/// - `cancellation`: Cancellation signal for this provider request.
///
/// # Returns
/// The provider assistant event stream. Cancellation and premature provider
/// stream completion are represented as terminal events.
pub fn stream_provider_cancellable(
    provider: Arc<dyn Provider>,
    model: Model,
    context: Context,
    mut options: SimpleStreamOptions,
    request_secret: Option<(RequestSecret, bool)>,
    applied_settings: Option<AppliedRequestSettingsCallback>,
    continuation: Option<ProviderContinuation>,
    continuation_enabled: bool,
    cancellation: CancellationToken,
) -> AssistantMessageEventStream {
    let (mut producer, stream) = create_assistant_message_event_stream();
    options.api_key = None;
    options.signal = cancellation.clone();
    tokio::spawn(async move {
        let provider_id = provider.id().to_owned();
        let api = model.api.clone();
        let model_id = model.id.clone();
        let call = async {
            if continuation_enabled {
                let callback = applied_settings.unwrap_or_else(|| Arc::new(|_| {}));
                provider
                    .stream_simple_with_continuation_and_applied_settings(
                        &model,
                        &context,
                        continuation.as_ref(),
                        &options,
                        request_secret
                            .as_ref()
                            .map(|(secret, bearer)| (secret, *bearer)),
                        callback,
                    )
                    .await
            } else if continuation.is_some() {
                continuation_error_stream(
                    &model,
                    &provider_id,
                    "provider does not support response continuation",
                )
            } else {
                match (request_secret, applied_settings) {
                    (Some((secret, bearer)), Some(callback)) => {
                        provider
                            .stream_simple_with_request_secret_and_applied_settings(
                                &model, &context, &options, &secret, bearer, callback,
                            )
                            .await
                    }
                    (Some((secret, bearer)), None) => {
                        provider
                            .stream_simple_with_request_secret(
                                &model, &context, &options, &secret, bearer,
                            )
                            .await
                    }
                    (None, Some(callback)) => {
                        provider
                            .stream_simple_with_applied_settings(
                                &model, &context, &options, callback,
                            )
                            .await
                    }
                    (None, None) => provider.stream_simple(&model, &context, &options).await,
                }
            }
        };
        let mut provider_stream = tokio::select! {
            _ = cancellation.cancelled() => {
                producer.push(AssistantMessageEvent::Error {
                    reason: ErrorReason::Aborted,
                    error: AssistantMessage::terminal(
                        api,
                        provider_id,
                        model_id,
                        StopReason::Aborted,
                        "provider request cancelled",
                        0,
                    ),
                });
                return;
            }
            stream = call => stream,
        };
        let mut terminal = false;
        loop {
            let event = tokio::select! {
                _ = cancellation.cancelled() => {
                    producer.push(AssistantMessageEvent::Error {
                        reason: ErrorReason::Aborted,
                        error: AssistantMessage::terminal(
                            api.clone(),
                            provider_id.clone(),
                            model_id.clone(),
                            StopReason::Aborted,
                            "provider request cancelled",
                            0,
                        ),
                    });
                    return;
                }
                event = provider_stream.next() => event,
            };
            let Some(event) = event else { break };
            terminal = event.is_terminal();
            if !producer.push(event) || terminal {
                return;
            }
        }
        if !terminal {
            producer.push(AssistantMessageEvent::Error {
                reason: ErrorReason::Error,
                error: AssistantMessage::terminal(
                    api,
                    provider_id,
                    model_id,
                    StopReason::Error,
                    "provider stream ended without a terminal event",
                    0,
                ),
            });
        }
    });
    stream
}

/// Continue a response the provider deferred (a long-poll or batch API).
///
/// Mirrors the optional `streamDeferred` / `cancelDeferred` on upstream's models
/// interface. Same contract as [`Provider::stream_simple`]: returns the consumer
/// end of a stream and reports failures as `Error` events rather than `Err`, so
/// the harness handles both kinds of provider call uniformly.
#[async_trait::async_trait]
pub trait DeferredProvider: Send + Sync {
    /// Poll a handle this provider previously issued: either the finished
    /// assistant message, or another `Deferred` handle to poll again.
    async fn stream_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        opts: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream;

    /// Best-effort remote cancellation. Defaulting to `Ok` is right because most
    /// long-poll APIs cannot be cancelled, and a caller must never be blocked by
    /// that.
    async fn cancel_deferred(
        &self,
        _model: &Model,
        _handle: &DeferredHandle,
    ) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod cancellable_stream_tests {
    use super::*;
    use crate::providers::faux::{FauxProvider, FauxScript};

    struct SilentProvider {
        models: Vec<Model>,
    }

    #[async_trait::async_trait]
    impl Provider for SilentProvider {
        fn id(&self) -> &str {
            "silent"
        }

        fn models(&self) -> &[Model] {
            &self.models
        }

        async fn stream_simple(
            &self,
            _model: &Model,
            _context: &Context,
            _options: &SimpleStreamOptions,
        ) -> AssistantMessageEventStream {
            let (_producer, stream) = create_assistant_message_event_stream();
            stream
        }
    }

    #[tokio::test]
    async fn cancelled_provider_stream_emits_one_terminal_abort() {
        let provider = FauxProvider::new(FauxScript::new().with_text("hello"));
        let model = provider.models()[0].clone();
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let mut stream = stream_provider_cancellable(
            provider,
            model,
            Context::default(),
            SimpleStreamOptions::default(),
            None,
            None,
            None,
            false,
            cancellation,
        );

        let event = stream.next().await.expect("abort event");
        assert!(event.is_terminal());
        assert!(matches!(
            event,
            AssistantMessageEvent::Error { error, .. }
                if error.stop_reason == StopReason::Aborted
        ));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn unterminated_provider_stream_gets_terminal_error() {
        let source = FauxProvider::new(FauxScript::new());
        let provider = Arc::new(SilentProvider {
            models: vec![source.models()[0].clone()],
        });
        let model = provider.models()[0].clone();
        let mut stream = stream_provider_cancellable(
            provider,
            model,
            Context::default(),
            SimpleStreamOptions::default(),
            None,
            None,
            None,
            false,
            CancellationToken::new(),
        );

        let event = stream.next().await.expect("repaired terminal event");
        assert!(matches!(
            event,
            AssistantMessageEvent::Error { error, .. }
                if error.stop_reason == StopReason::Error
        ));
        assert!(stream.next().await.is_none());
    }
}

// ---------------------------------------------------------------------------
// ProviderHooks (B4) — a per-call sidecar hook fired inside the harness
// `StreamFn` closure BEFORE each `stream_simple` call.
//
// Why a sidecar trait (not an extension of `Provider`): `Provider` is not
// sealed (verified), so extending it directly is viable — but that would force
// EVERY `Provider` impl to stub no-op hooks, even faux/anthropic. A sidecar
// trait keeps `Provider` lean and lets only the host (harness) supply hooks;
// the harness captures `Option<Arc<dyn ProviderHooks>>` into its `StreamFn`
// closure and applies the patch per call. This mirrors pi's `beforeRequest`
// per-call semantics, NOT a run-once config-build patch (applying once per run
// is not equivalent — e.g. a hook that injects a per-request id or rotates a
// header must run before every provider call).
//
// The patch type is `SimpleStreamOptionsPatch` (against `SimpleStreamOptions`,
// the type actually passed to `stream_simple`) — NOT the harness-side
// `AgentHarnessStreamOptionsPatch` (which targets `AgentHarnessStreamOptions`,
// the pinned subset, and lives in rpi-harness): defining it here keeps rpi-ai
// free of any rpi-harness dep (cycle-free) and targets the live options the
// provider sees. The harness's existing `AgentHarnessStreamOptionsPatch` stays
// for harness config, a separate concern.
// ---------------------------------------------------------------------------

/// A per-call provider hook. `before_request` may patch the live
/// [`SimpleStreamOptions`] (e.g. inject/rotate headers, set metadata, override
/// timeout) before the harness calls `stream_simple`; `after_response` observes
/// the terminal assistant message after the stream resolves. Both are
/// `Option`-returning: `None` = no change / no observation.
///
/// Implementations MUST be infallible at the API boundary — a hook error is
/// swallowed (the run continues with the unpatched options), mirroring the
/// "hooks must not throw" contract from pi and the harness's other hooks
/// (`before_tool_call` etc.).
pub trait ProviderHooks: Send + Sync {
    /// Fired before each `stream_simple` call, with a snapshot of the model +
    /// context + opts the harness is about to pass. Return a patch that the
    /// harness applies to a clone of `ctx` before calling the provider; return
    /// `None` to leave `ctx` untouched.
    ///
    /// This is the seam pi calls `transform_context`
    /// (`harness/execution/assistant.ts:142`), and it fires on **every** provider
    /// request of a run — the same cadence as [`Self::before_request`], and the
    /// reason a plugin can change the prompt for one turn without the change
    /// sticking to the session. It runs before `before_request` so a prompt the
    /// hook installs is what that request carries.
    ///
    /// Unlike `before_request` this cannot patch `opts`; the two are separate so
    /// a caller can reason about "what the model sees" independently of "how the
    /// request is sent".
    ///
    /// Must not retain the borrow. Runs on the blocking-bridge thread inside the
    /// harness `StreamFn` closure.
    fn transform_context(&self, _model: &Model, _ctx: &Context) -> Option<ContextPatch> {
        None
    }

    /// Fired before each `stream_simple` call, with a snapshot of the model +
    /// context + opts the harness is about to pass. Return a patch that the
    /// harness applies to a clone of `opts` before calling the provider; return
    /// `None` to leave `opts` untouched.
    ///
    /// The hook receives a reference to the planned `opts`, not ownership — it
    /// must not retain the borrow (the closure clones before `await`). Runs on
    /// the blocking-bridge thread inside the harness `StreamFn` closure.
    fn before_request(
        &self,
        _model: &Model,
        _ctx: &Context,
        _opts: &SimpleStreamOptions,
    ) -> Option<SimpleStreamOptionsPatch> {
        None
    }

    /// Fired after the stream resolves with the terminal assistant message.
    /// `message` is the final `AssistantMessage` the provider produced (the same
    /// one the loop emits as `MessageEnd`). `None` lets the default no-op impl
    /// apply. Runs on the blocking-bridge thread.
    fn after_response(&self, _model: &Model, _message: &AssistantMessage) {}
}

/// A no-op default so the harness can install a `ProviderHooks` slot cheaply
/// when none is supplied (uniform code path vs `Option<Arc<dyn>>`).
#[derive(Default)]
pub struct NoopProviderHooks;
impl ProviderHooks for NoopProviderHooks {}

/// A patch against the model-facing [`Context`] returned by
/// [`ProviderHooks::transform_context`].
///
/// Mirrors the `systemPrompt` half of pi's `transform_context` result
/// (`harness/agent-harness.ts:445`): the hook may replace the system prompt the
/// request carries. pi also allows replacing `messages` there; rpi deliberately
/// does not (yet) — see `ProviderHooks::transform_context`.
///
/// `None` inside means "leave as-is", so an empty patch is a no-op and a caller
/// that only wants to observe can return `None` from the hook entirely.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextPatch {
    /// Replacement system prompt for this request. `Some` installs it verbatim
    /// (**replacement, not append** — a plugin that wants to append reads
    /// `ctx.system_prompt` and concatenates, exactly as pi requires).
    pub system_prompt: Option<String>,
}

impl ContextPatch {
    /// Apply this patch to `ctx` in place.
    pub fn apply(&self, ctx: &mut Context) {
        if let Some(prompt) = &self.system_prompt {
            ctx.system_prompt = Some(prompt.clone());
        }
    }
}

/// A patch against [`SimpleStreamOptions`] returned by [`ProviderHooks::before_request`].
/// Mirrors TS `beforeRequest`'s per-request overrides but targets the live
/// `SimpleStreamOptions` (the type `stream_simple` receives), not the harness
/// pinned subset. `Option<Option<T>>` semantics: outer `None` = "leave this
/// field"; inner `Some(None)` on the headers/metadata maps = "delete this key".
///
/// This is *distinct* from the harness-side `AgentHarnessStreamOptionsPatch`
/// (which lives in rpi-harness and patches `AgentHarnessStreamOptions`). Both
/// exist; this one is the rpi-ai seam so provider hooks can be defined without
/// a rpi-harness cycle.
#[derive(Debug, Clone, Default)]
pub struct SimpleStreamOptionsPatch {
    pub timeout: Option<Option<Duration>>,
    pub max_retries: Option<Option<u32>>,
    pub max_retry_delay: Option<Option<Duration>>,
    /// `None` = leave; `Some(map)` = merge, inner `None` deletes a key.
    pub headers: Option<BTreeMap<String, Option<String>>>,
    /// `None` = leave; `Some(map)` = merge, inner `None` deletes a key.
    pub metadata: Option<BTreeMap<String, Option<String>>>,
    pub cache_retention: Option<Option<CacheRetention>>,
    pub max_tokens: Option<Option<u64>>,
    pub temperature: Option<Option<f64>>,
    pub session_id: Option<Option<String>>,
}

impl SimpleStreamOptionsPatch {
    /// Apply this patch to `opts` in place. Mirrors the harness patch semantics:
    /// outer `None` = leave the field; an explicit value replaces it; `None`
    /// inside a map entry deletes that key.
    pub fn apply(&self, opts: &mut SimpleStreamOptions) {
        if let Some(v) = self.timeout {
            opts.timeout = v;
        }
        if let Some(v) = self.max_retries {
            opts.max_retries = v;
        }
        if let Some(v) = self.max_retry_delay {
            opts.max_retry_delay = v;
        }
        if let Some(patch) = &self.headers {
            let mut map = opts.headers.take().unwrap_or_default();
            for (k, v) in patch {
                match v {
                    Some(val) => map.insert(k.clone(), val.clone()),
                    None => map.remove(k),
                };
            }
            opts.headers = if map.is_empty() { None } else { Some(map) };
        }
        if let Some(patch) = &self.metadata {
            let mut map = opts.metadata.take().unwrap_or_default();
            for (k, v) in patch {
                match v {
                    Some(val) => map.insert(k.clone(), val.clone()),
                    None => map.remove(k),
                };
            }
            opts.metadata = if map.is_empty() { None } else { Some(map) };
        }
        if let Some(v) = self.cache_retention {
            opts.cache_retention = v.unwrap_or_default();
        }
        if let Some(v) = self.max_tokens {
            opts.max_tokens = v;
        }
        if let Some(v) = self.temperature {
            opts.temperature = v;
        }
        if let Some(v) = &self.session_id {
            opts.session_id = v.clone();
        }
    }
}
