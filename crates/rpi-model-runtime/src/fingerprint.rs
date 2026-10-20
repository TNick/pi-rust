//! Builds conservative, secret-free model execution fingerprints.

use rpi_ai::{Model, StreamingProtocolCompat};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const MODEL_EXECUTION_SCHEMA: &str = "rpi-model-execution-v1";

/// Describes the legacy model-configuration digest coverage.
///
/// # Public members
/// - `sha256`: Digest from the conservative execution projection.
/// - `headers_excluded`: Always true; request headers are omitted.
/// - `provider_options_included`: Always false; opaque options are omitted.
///
/// # Private members
/// None.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelConfigurationFingerprint {
    /// SHA-256 digest of the conservative typed model projection.
    pub sha256: String,
    /// Whether resolved request headers were excluded.
    pub headers_excluded: bool,
    /// Whether runtime-only provider options were included.
    pub provider_options_included: bool,
}

/// Identifies a model field omitted from a safe execution fingerprint.
///
/// # Public members
/// - `Endpoint`: Endpoint strings are omitted because they are unvalidated.
/// - `Headers`: Request headers may contain credentials.
/// - `ProviderOptions`: Runtime-only options are opaque and may be sensitive.
/// - `SamplingParams`: Arbitrary request values are not validated as public.
/// - `ThinkingLevelMap`: Provider-specific values are unconstrained strings.
/// - `OpaqueCompatibility`: Custom compatibility metadata is arbitrary JSON.
///
/// # Private members
/// None.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelExecutionExclusion {
    /// The model endpoint may contain user information or secret query values.
    Endpoint,
    /// Resolved request headers may contain credentials or tokens.
    Headers,
    /// Runtime-only opaque provider options may contain sensitive values.
    ProviderOptions,
    /// Arbitrary sampling values have no validated non-secret contract.
    SamplingParams,
    /// Provider-specific thinking labels are unconstrained strings.
    ThinkingLevelMap,
    /// Custom compatibility metadata is arbitrary JSON.
    OpaqueCompatibility,
}

/// Safe identifiers and coverage metadata for one model configuration hash.
///
/// The digest includes typed public model identity and known typed metadata.
/// Omitted values are reported, and `complete` is false whenever one of those
/// values could change the destination or request. The provider and model IDs
/// are treated as public identifiers; callers must not place credentials in
/// those fields.
///
/// # Public members
/// - `schema_version`: Fingerprint schema identifier.
/// - `provider_id`: Public provider identifier from the model catalog.
/// - `model_id`: Public model identifier from the model catalog.
/// - `api_id`: Public provider API identifier.
/// - `sha256`: Digest of the canonical safe typed projection.
/// - `complete`: Whether omitted values cannot affect execution.
/// - `excluded_fields`: Present excluded fields that limit coverage.
/// - `headers_excluded`: Always true; request headers never enter the digest.
/// - `provider_options_included`: Always false; opaque options never enter it.
///
/// # Private members
/// None.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelExecutionFingerprint {
    /// Version of the canonical projection hashed by this API.
    pub schema_version: String,
    /// Public provider identifier from the model catalog.
    pub provider_id: String,
    /// Public model identifier from the model catalog.
    pub model_id: String,
    /// Public API identifier from the model descriptor.
    pub api_id: String,
    /// Lowercase SHA-256 digest of the canonical safe typed projection.
    pub sha256: String,
    /// Whether omitted values cannot affect the model execution.
    pub complete: bool,
    /// Present fields omitted because their safety or semantics are unknown.
    pub excluded_fields: Vec<ModelExecutionExclusion>,
    /// Whether request headers were excluded from the projection.
    pub headers_excluded: bool,
    /// Whether runtime-only provider options were included in the digest.
    pub provider_options_included: bool,
}

/// Fingerprints a conservative typed projection of upstream model metadata.
///
/// Stable public IDs and typed fields are included. Endpoint strings, headers,
/// runtime-only options, arbitrary sampling JSON, unconstrained thinking maps,
/// and opaque compatibility JSON are excluded. Since endpoint identity is
/// always omitted, this API always reports incomplete coverage. A future full
/// fingerprint requires validated endpoint identity and provider-owned safe
/// projections for currently opaque request-shaping fields.
///
/// # Parameters
/// - `model`: Resolved upstream model descriptor to fingerprint.
///
/// # Returns
/// A versioned digest with explicit excluded-field and coverage metadata.
pub fn model_execution_fingerprint(model: &Model) -> ModelExecutionFingerprint {
    let mut projection = Map::new();
    projection.insert(
        "schemaVersion".into(),
        Value::String(MODEL_EXECUTION_SCHEMA.into()),
    );
    projection.insert("providerId".into(), Value::String(model.provider.clone()));
    projection.insert("modelId".into(), Value::String(model.id.clone()));
    projection.insert("apiId".into(), Value::String(model.api.as_str().into()));
    projection.insert("reasoning".into(), Value::Bool(model.reasoning));
    projection.insert(
        "input".into(),
        Value::Array(
            model
                .input
                .iter()
                .map(|modality| {
                    Value::String(
                        match modality {
                            rpi_ai::InputModality::Text => "text",
                            rpi_ai::InputModality::Image => "image",
                        }
                        .into(),
                    )
                })
                .collect(),
        ),
    );
    projection.insert(
        "cost".into(),
        serde_json::to_value(&model.cost).expect("model cost serialization is infallible"),
    );
    projection.insert("costConfigured".into(), Value::Bool(model.cost_configured));
    projection.insert("contextWindow".into(), Value::from(model.context_window));
    projection.insert("maxTokens".into(), Value::from(model.max_tokens));

    let mut excluded_fields = vec![ModelExecutionExclusion::Endpoint];
    if model
        .headers
        .as_ref()
        .is_some_and(|headers| !headers.is_empty())
    {
        excluded_fields.push(ModelExecutionExclusion::Headers);
    }
    if !model.provider_options.is_empty() {
        excluded_fields.push(ModelExecutionExclusion::ProviderOptions);
    }
    if model.sampling_params.is_some() {
        excluded_fields.push(ModelExecutionExclusion::SamplingParams);
    }
    if model.thinking_level_map.is_some() {
        excluded_fields.push(ModelExecutionExclusion::ThinkingLevelMap);
    }
    match model.compat.as_ref() {
        Some(StreamingProtocolCompat::Other(_)) => {
            excluded_fields.push(ModelExecutionExclusion::OpaqueCompatibility);
        }
        Some(compat) => {
            projection.insert(
                "compat".into(),
                serde_json::to_value(compat).expect("typed compatibility serialization"),
            );
        }
        None => {
            projection.insert("compat".into(), Value::Null);
        }
    }

    canonicalize(&mut projection);
    let bytes = serde_json::to_vec(&Value::Object(projection))
        .expect("safe model projection serialization is infallible");
    let digest = Sha256::digest(bytes);
    ModelExecutionFingerprint {
        schema_version: MODEL_EXECUTION_SCHEMA.into(),
        provider_id: model.provider.clone(),
        model_id: model.id.clone(),
        api_id: model.api.as_str().into(),
        sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
        complete: excluded_fields.is_empty(),
        excluded_fields,
        headers_excluded: true,
        provider_options_included: false,
    }
}

/// Fingerprints model configuration using the conservative execution schema.
///
/// This compatibility name delegates to [`model_execution_fingerprint`]. The
/// returned digest is incomplete whenever excluded request-shaping metadata is
/// present; callers must inspect `complete` and `excluded_fields`.
///
/// # Parameters
/// - `model`: Resolved upstream model descriptor to fingerprint.
///
/// # Returns
/// The versioned execution fingerprint and its coverage metadata.
#[deprecated(note = "use model_execution_fingerprint to inspect coverage")]
pub fn model_configuration_fingerprint(model: &Model) -> ModelConfigurationFingerprint {
    let fingerprint = model_execution_fingerprint(model);
    ModelConfigurationFingerprint {
        sha256: fingerprint.sha256,
        headers_excluded: fingerprint.headers_excluded,
        provider_options_included: fingerprint.provider_options_included,
    }
}

/// Sorts JSON object keys recursively to stabilize the canonical encoding.
///
/// # Parameters
/// - `object`: JSON object whose keys should be ordered.
fn canonicalize(object: &mut Map<String, Value>) {
    let mut entries = object
        .iter_mut()
        .map(|(key, value)| {
            if let Value::Object(nested) = value {
                canonicalize(nested);
            }
            (key.clone(), value.clone())
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    object.clear();
    object.extend(entries);
}

#[cfg(test)]
mod tests {
    use rpi_ai::model::OpenaiCompletionsCompat;
    use rpi_ai::{Api, Model, StreamingProtocolCompat};
    use serde_json::json;

    use super::{model_execution_fingerprint, ModelExecutionExclusion as Exclusion};

    /// Creates a model with stable public identifiers and typed metadata.
    fn model() -> Model {
        let mut model = Model::new(
            "model-public-id",
            "Model display name",
            Api::OpenaiCompletions,
            "provider-public-id",
            "https://example.invalid/v1",
        );
        model.reasoning = true;
        model.context_window = 32_000;
        model.max_tokens = 4_096;
        model.compat = Some(StreamingProtocolCompat::OpenaiCompletions(
            OpenaiCompletionsCompat {
                supports_strict_mode: Some(true),
                ..OpenaiCompletionsCompat::default()
            },
        ));
        model
    }

    /// Exposes stable IDs and fingerprints included typed request metadata.
    #[test]
    fn includes_public_ids_and_typed_metadata() {
        let original = model();
        let original_fingerprint = model_execution_fingerprint(&original);
        let mut changed = original.clone();
        changed.max_tokens += 1;

        let changed_fingerprint = model_execution_fingerprint(&changed);

        assert_eq!(original_fingerprint.provider_id, "provider-public-id");
        assert_eq!(original_fingerprint.model_id, "model-public-id");
        assert_eq!(original_fingerprint.api_id, "openai-completions");
        assert_ne!(original_fingerprint.sha256, changed_fingerprint.sha256);
        assert!(!original_fingerprint.complete);
        assert!(original_fingerprint
            .excluded_fields
            .contains(&Exclusion::Endpoint));
    }

    /// Excludes endpoints, auth headers, runtime options, and opaque values.
    #[test]
    fn omits_untrusted_and_secret_bearing_values() {
        let mut original = model();
        original.base_url = "https://user:secret@example.invalid/path?token=one".into();
        original.headers = Some(std::collections::BTreeMap::from([(
            "authorization".into(),
            "header-secret-one".into(),
        )]));
        original
            .provider_options
            .insert("runtime-secret".into(), json!("runtime-secret-one"));
        original.sampling_params = Some(serde_json::Map::from_iter([(
            "custom".into(),
            json!("sampling-secret-one"),
        )]));
        original.thinking_level_map = Some(std::collections::BTreeMap::from([(
            rpi_ai::ThinkingLevel::High,
            Some("thinking-secret-one".into()),
        )]));
        original.compat = Some(StreamingProtocolCompat::Other(json!({
            "opaque": "compat-secret-one"
        })));
        let mut changed = original.clone();
        changed.base_url = "https://user:secret@example.invalid/path?token=two".into();
        changed
            .headers
            .as_mut()
            .unwrap()
            .insert("authorization".into(), "header-secret-two".into());
        changed
            .provider_options
            .insert("runtime-secret".into(), json!("runtime-secret-two"));
        changed
            .sampling_params
            .as_mut()
            .unwrap()
            .insert("custom".into(), json!("sampling-secret-two"));
        changed.thinking_level_map.as_mut().unwrap().insert(
            rpi_ai::ThinkingLevel::High,
            Some("thinking-secret-two".into()),
        );
        changed.compat = Some(StreamingProtocolCompat::Other(json!({
            "opaque": "compat-secret-two"
        })));

        let first = model_execution_fingerprint(&original);
        let second = model_execution_fingerprint(&changed);

        assert_eq!(first.sha256, second.sha256);
        assert!(!first.complete);
        for field in [
            Exclusion::Endpoint,
            Exclusion::Headers,
            Exclusion::ProviderOptions,
            Exclusion::SamplingParams,
            Exclusion::ThinkingLevelMap,
            Exclusion::OpaqueCompatibility,
        ] {
            assert!(first.excluded_fields.contains(&field));
        }
        let encoded = serde_json::to_string(&first).expect("fingerprint JSON");
        assert!(!encoded.contains("secret-one"));
        assert!(!encoded.contains("user:"));
    }

    /// Includes typed compatibility flags while unknown values remain omitted.
    #[test]
    fn includes_typed_compatibility_metadata() {
        let mut original = model();
        let mut changed = original.clone();
        original.compat = Some(StreamingProtocolCompat::OpenaiCompletions(
            OpenaiCompletionsCompat {
                supports_strict_mode: Some(true),
                ..OpenaiCompletionsCompat::default()
            },
        ));
        changed.compat = Some(StreamingProtocolCompat::OpenaiCompletions(
            OpenaiCompletionsCompat {
                supports_strict_mode: Some(false),
                ..OpenaiCompletionsCompat::default()
            },
        ));

        assert_ne!(
            model_execution_fingerprint(&original).sha256,
            model_execution_fingerprint(&changed).sha256
        );
    }
}
