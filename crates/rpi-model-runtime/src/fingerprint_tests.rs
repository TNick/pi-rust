//! Tests safe model execution fingerprint coverage and secret omission.

use rpi_ai::model::OpenaiCompletionsCompat;
use rpi_ai::{Api, Model, ProviderOptionsSchema, StreamingProtocolCompat};
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
    assert!(original_fingerprint.complete);
    assert!(original_fingerprint.endpoint_sha256.is_some());
    assert!(!original_fingerprint
        .excluded_fields
        .contains(&Exclusion::Endpoint));
}

/// Uses endpoint identity without retaining its raw value.
#[test]
fn safe_endpoint_changes_fingerprint_without_exposing_endpoint() {
    let original = model();
    let mut changed = original.clone();
    changed.base_url = "https://gateway.example.invalid/v2".into();

    let first = model_execution_fingerprint(&original);
    let second = model_execution_fingerprint(&changed);

    assert_ne!(first.sha256, second.sha256);
    assert_ne!(first.endpoint_sha256, second.endpoint_sha256);
    let encoded = serde_json::to_string(&first).expect("fingerprint JSON");
    assert!(!encoded.contains(&original.base_url));
}

/// Includes recognized sampling values in the opaque execution identity.
#[test]
fn sampling_configuration_changes_fingerprint_without_becoming_excluded() {
    let mut original = model();
    original.sampling_params = Some(serde_json::Map::from_iter([(
        "temperature".into(),
        json!(0.2),
    )]));
    let mut changed = original.clone();
    changed
        .sampling_params
        .as_mut()
        .expect("sampling parameters")
        .insert("temperature".into(), json!(0.7));

    let first = model_execution_fingerprint(&original);
    let second = model_execution_fingerprint(&changed);

    assert_ne!(first.sha256, second.sha256);
    assert!(first.complete);
    assert!(!first.excluded_fields.contains(&Exclusion::SamplingParams));
}

/// Includes schema-safe provider options in the opaque execution identity.
#[test]
fn safe_provider_option_changes_fingerprint_without_exposing_its_value() {
    let mut original = model();
    original.provider_options_schema = Some(ProviderOptionsSchema {
        version: 1,
        schema: json!({
            "type": "object",
            "properties": {
                "future_vendor_tuning": {
                    "type": "string",
                    "enum": ["fast", "accurate"]
                }
            },
            "additionalProperties": false
        }),
    });
    original
        .provider_options
        .insert("future_vendor_tuning".into(), json!("fast"));
    let mut changed = original.clone();
    changed
        .provider_options
        .insert("future_vendor_tuning".into(), json!("accurate"));

    let first = model_execution_fingerprint(&original);
    let second = model_execution_fingerprint(&changed);

    assert_ne!(first.sha256, second.sha256);
    assert!(first.provider_options_included);
    assert!(!first.excluded_fields.contains(&Exclusion::ProviderOptions));
    let encoded = serde_json::to_string(&first).expect("fingerprint JSON");
    assert!(!encoded.contains("fast"));
}

/// Excludes malformed options from the opaque execution identity.
#[test]
fn invalid_provider_option_values_are_counted_but_not_fingerprinted() {
    let mut original = model();
    original.provider_options_schema = Some(ProviderOptionsSchema {
        version: 1,
        schema: json!({
            "type": "object",
            "properties": {
                "vendor_mode": {"type": "string", "enum": ["fast", "accurate"]}
            },
            "additionalProperties": false
        }),
    });
    original
        .provider_options
        .insert("vendor_mode".into(), json!("invalid-one"));
    let mut changed = original.clone();
    changed
        .provider_options
        .insert("vendor_mode".into(), json!("invalid-two"));

    let first = model_execution_fingerprint(&original);
    let second = model_execution_fingerprint(&changed);

    assert_eq!(first.sha256, second.sha256);
    assert!(!first.provider_options_included);
    assert!(first.excluded_fields.contains(&Exclusion::ProviderOptions));
    let encoded = serde_json::to_string(&first).expect("fingerprint JSON");
    assert!(!encoded.contains("invalid-one"));
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
