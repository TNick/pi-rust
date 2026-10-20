//! Tests public model snapshots and safe execution-configuration projections.

use std::collections::BTreeMap;

use rpi_ai::model::OpenaiCompletionsCompat;
use rpi_ai::{Api, Model, StreamingProtocolCompat, ThinkingLevel};
use serde_json::json;

use super::{SafeModelCompatibility, safe_model_execution_configuration, safe_model_snapshot};
use crate::ModelExecutionExclusion as Exclusion;

/// Builds a model containing useful public metadata.
fn model() -> Model {
    let mut model = Model::new(
        "model-id",
        "Display name",
        Api::OpenaiCompletions,
        "provider-id",
        "https://example.invalid/v1",
    );
    model.reasoning = true;
    model.context_window = 32_000;
    model.max_tokens = 4_096;
    model.cost_configured = true;
    model.thinking_level_map = Some(BTreeMap::from([
        (ThinkingLevel::Off, None),
        (ThinkingLevel::High, Some("effort-secret".to_owned())),
        (ThinkingLevel::Xhigh, Some("xhigh-secret".to_owned())),
    ]));
    model.compat = Some(StreamingProtocolCompat::OpenaiCompletions(
        OpenaiCompletionsCompat {
            supports_strict_mode: Some(true),
            ..OpenaiCompletionsCompat::default()
        },
    ));
    model
}

/// Emits typed display metadata and the model's upstream thinking levels.
#[test]
fn projects_safe_typed_metadata() {
    let snapshot = safe_model_snapshot(&model());

    assert_eq!(snapshot.schema_version, "rpi-safe-model-snapshot-v1");
    assert_eq!(snapshot.provider_id, "provider-id");
    assert_eq!(snapshot.model_id, "model-id");
    assert_eq!(snapshot.display_name, "Display name");
    assert_eq!(snapshot.api_id, "openai-completions");
    assert!(snapshot.reasoning);
    assert!(snapshot.cost_configured);
    assert_eq!(snapshot.context_window, 32_000);
    assert_eq!(snapshot.max_tokens, 4_096);
    assert_eq!(
        snapshot.thinking_levels,
        vec![
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::Xhigh,
        ]
    );
    assert!(matches!(
        snapshot.compatibility,
        Some(SafeModelCompatibility::OpenaiCompletions(_))
    ));
}

/// Omits secret-bearing and arbitrary fields while reporting their presence.
#[test]
fn excludes_execution_fields_and_reports_coverage() {
    let mut model = model();
    model.base_url = "https://user:credential@example.invalid/?token=secret".into();
    model.headers = Some(BTreeMap::from([(
        "authorization".into(),
        "header-secret".into(),
    )]));
    model.sampling_params = Some(serde_json::Map::from_iter([(
        "custom".into(),
        json!("sampling-secret"),
    )]));
    model
        .provider_options
        .insert("private-option".into(), json!("option-secret"));
    model.compat = Some(StreamingProtocolCompat::Other(json!({
        "secret": "opaque-compat-secret"
    })));

    let snapshot = safe_model_snapshot(&model);
    let encoded = serde_json::to_string(&snapshot).expect("serialize snapshot");

    for secret in [
        "credential",
        "token=secret",
        "header-secret",
        "sampling-secret",
        "option-secret",
        "opaque-compat-secret",
    ] {
        assert!(!encoded.contains(secret));
    }
    for field in [
        Exclusion::Endpoint,
        Exclusion::Headers,
        Exclusion::ProviderOptions,
        Exclusion::SamplingParams,
        Exclusion::ThinkingLevelMap,
        Exclusion::OpaqueCompatibility,
    ] {
        assert!(snapshot.excluded_execution_fields.contains(&field));
    }
    assert!(snapshot.compatibility.is_none());
}

/// Includes only allowlisted numeric sampling values in provenance output.
#[test]
fn execution_configuration_allowlists_sampling_values() {
    let mut model = model();
    model.sampling_params = Some(serde_json::Map::from_iter([
        ("temperature".into(), json!(0.25)),
        ("top_p".into(), json!(0.8)),
        ("seed".into(), json!(42)),
        ("max_tokens".into(), json!(2048)),
        ("user".into(), json!("user-secret")),
        ("extension".into(), json!({ "secret": "opaque-secret" })),
    ]));
    model.base_url = "https://user:credential@example.invalid/?token=secret".into();
    model.headers = Some(BTreeMap::from([(
        "authorization".into(),
        "header-secret".into(),
    )]));
    model
        .provider_options
        .insert("private-option".into(), json!("option-secret"));

    let configuration = safe_model_execution_configuration(&model);
    let encoded = serde_json::to_string(&configuration).expect("serialize config");

    assert_eq!(configuration.schema_version, "rpi-safe-model-execution-v1");
    assert_eq!(configuration.sampling_parameters.temperature, Some(0.25));
    assert_eq!(configuration.sampling_parameters.top_p, Some(0.8));
    assert_eq!(configuration.sampling_parameters.seed, Some(42));
    assert_eq!(configuration.sampling_parameters.max_tokens, Some(2048));
    assert_eq!(configuration.excluded_sampling_parameter_count, 2);
    assert!(!configuration.complete);
    for excluded in [
        Exclusion::Endpoint,
        Exclusion::Headers,
        Exclusion::ProviderOptions,
        Exclusion::SamplingParams,
        Exclusion::ThinkingLevelMap,
    ] {
        assert!(configuration.excluded_execution_fields.contains(&excluded));
    }
    for secret in [
        "credential",
        "token=secret",
        "header-secret",
        "option-secret",
        "user-secret",
        "opaque-secret",
    ] {
        assert!(!encoded.contains(secret));
    }
    assert!(!encoded.contains("extension"));
    assert!(!encoded.contains("user\""));
}
