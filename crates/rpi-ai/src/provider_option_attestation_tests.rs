//! Regression tests for secret-free provider-option application evidence.

use crate::{Api, Model, ProviderOptionsSchema};
use serde_json::{json, Value};
use std::collections::BTreeMap;

use super::provider_option_attestation::{
    attest_provider_options, ProviderOptionDisposition as Disposition,
};

/// Creates a model with public enum and sensitive schema properties.
fn model() -> Model {
    let mut model = Model::new(
        "model-a",
        "Model A",
        Api::OpenaiCompletions,
        "provider-a",
        "https://example.invalid",
    );
    model.provider_options_schema = Some(ProviderOptionsSchema {
        version: 1,
        schema: json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "careful"]},
                "access_token": {"type": "string", "enum": ["x", "y"]},
                "nested": {"type": "object", "properties": {"x": {"type": "string"}}}
            },
            "additionalProperties": false
        }),
    });
    model
}

/// Distinguishes applied, transformed, and ignored values without exposing them.
#[test]
fn attests_applied_transformed_and_ignored_options_opaquely() {
    let mut model = model();
    model.provider_options.insert("mode".into(), json!("fast"));
    let effective = BTreeMap::from([("mode".to_owned(), json!("careful"))]);

    let report = attest_provider_options(&model, &effective);
    let option = &report.options[0];
    assert_eq!(option.disposition, Disposition::Transformed);
    assert!(option.effective_value_sha256.is_some());
    let ignored = attest_provider_options(&model, &BTreeMap::new());
    assert_eq!(ignored.options[0].disposition, Disposition::Ignored);
    assert!(ignored.options[0].effective_value_sha256.is_none());
    let encoded = serde_json::to_string(&report).expect("report JSON");
    assert!(!encoded.contains("mode"));
    assert!(!encoded.contains("careful"));
}

/// Omits credential-named and composite values from option evidence.
#[test]
fn omits_sensitive_and_composite_options() {
    let mut model = model();
    model.provider_options.insert("mode".into(), json!("fast"));
    model
        .provider_options
        .insert("access_token".into(), json!("x"));
    model
        .provider_options
        .insert("nested".into(), json!({"x":"y"}));
    assert!(model.validated_provider_options().is_ok());

    let (safe, omitted) = super::provider_option_attestation::safe_provider_options(&model);
    assert_eq!(safe.len(), 1);
    assert_eq!(omitted, 2);
    let report = attest_provider_options(&model, &BTreeMap::<String, Value>::new());
    assert_eq!(report.options.len(), 1);
    assert_eq!(report.omitted_count, 2);
    let encoded = serde_json::to_string(&report).expect("report JSON");
    assert!(!encoded.contains("\"x\""));
    assert!(!encoded.contains("access_token"));
}

/// Counts undeclared effective values without including their keys or values.
#[test]
fn counts_unrecognized_effective_options() {
    let report = attest_provider_options(
        &model(),
        &BTreeMap::from([("undeclared".to_owned(), json!("private"))]),
    );
    assert!(report.options.is_empty());
    assert_eq!(report.omitted_count, 1);
    let encoded = serde_json::to_string(&report).expect("report JSON");
    assert!(!encoded.contains("undeclared"));
    assert!(!encoded.contains("private"));
}

/// Omits a provider-transformed value that violates the safe scalar schema.
#[test]
fn omits_unverifiable_effective_values() {
    let mut model = model();
    model.provider_options.insert("mode".into(), json!("fast"));
    let report = attest_provider_options(
        &model,
        &BTreeMap::from([("mode".to_owned(), json!({"credential": "private"}))]),
    );

    assert_eq!(report.options[0].disposition, Disposition::Unverifiable);
    assert!(report.options[0].effective_value_sha256.is_none());
    assert_eq!(report.omitted_count, 1);
    let encoded = serde_json::to_string(&report).expect("report JSON");
    assert!(!encoded.contains("credential"));
    assert!(!encoded.contains("private"));
}
