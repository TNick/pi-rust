//! Regression fixtures for preserving opaque provider options as runtime data.

use super::*;
use crate::{model_property_descriptors, set_model_properties, set_model_property};
use rpi_ai::ProviderOptionsSchema;

/// Resolves literal test configuration without reading host environment values.
struct EmptyConfigResolver;

impl ConfigValueResolver for EmptyConfigResolver {
    /// Returns literal configuration values unchanged.
    fn resolve(
        &self,
        value: &str,
        _environment: Option<&BTreeMap<String, String>>,
    ) -> Option<String> {
        Some(value.to_owned())
    }

    /// Reports no process environment values.
    fn environment_value(&self, _name: &str) -> Option<String> {
        None
    }
}

/// Preserves unknown provider fields in config while keeping model projections safe.
#[test]
fn provider_options_round_trip_into_runtime_model_without_public_leakage() {
    let source = serde_json::json!({
        "providers": {
            "pluggy": {
                "api": "openai-completions",
                "baseUrl": "https://example.test",
                "models": [{
                    "id": "model-a",
                    "reasoning_effort": "opaque-sentinel-value",
                    "request_mode": {"variant": "opaque-sentinel-nested"}
                }]
            }
        }
    });
    let config: ModelsConfig = serde_json::from_value(source).expect("model configuration");
    let provider = config.providers.get("pluggy").expect("provider config");
    let definition = provider.models.first().expect("model definition");

    assert_eq!(
        definition.provider_options["reasoning_effort"],
        "opaque-sentinel-value"
    );
    let serialized = serde_json::to_value(&config).expect("serialized config");
    assert_eq!(
        serialized["providers"]["pluggy"]["models"][0]["request_mode"]["variant"],
        "opaque-sentinel-nested"
    );
    let restored: ModelsConfig = serde_json::from_value(serialized).expect("restored config");
    let restored_provider = restored.providers.get("pluggy").expect("restored provider");
    let model = provider_to_models("pluggy", restored_provider, &EmptyConfigResolver)
        .expect("supported provider models")
        .remove(0);

    assert_eq!(
        model.provider_options["reasoning_effort"],
        "opaque-sentinel-value"
    );
    assert_eq!(
        model.provider_options["request_mode"]["variant"],
        "opaque-sentinel-nested"
    );
    let public_model = serde_json::to_string(&model).expect("public model JSON");
    assert!(!public_model.contains("opaque-sentinel"));
    let debug_model = format!("{model:?}");
    assert!(!debug_model.contains("opaque-sentinel"));
}

/// Exposes only schema-backed properties and validates/coerces UI edits.
#[test]
fn schema_backed_model_property_editor_api() {
    let mut model = Model::new(
        "model-a",
        "Model A",
        Api::OpenaiCompletions,
        "pluggy",
        "https://example.test",
    );
    model.provider_options_schema = Some(ProviderOptionsSchema {
        version: 1,
        schema: serde_json::json!({
            "type": "object",
            "properties": {
                "vendor_mode": {
                    "type": "string",
                    "title": "Reasoning effort",
                    "enum": ["low", "high", "strict"],
                    "default": "low"
                },
                "provider_budget": {"type": "integer"}
            },
            "required": ["vendor_mode"],
            "additionalProperties": false
        }),
    });

    let descriptors = model_property_descriptors(&model).expect("schema properties");
    assert_eq!(descriptors.len(), 2);
    let effort = descriptors
        .iter()
        .find(|property| property.key == "vendor_mode")
        .expect("vendor mode property");
    assert!(effort.required);
    assert_eq!(effort.title.as_deref(), Some("Reasoning effort"));
    assert_eq!(effort.default, Some(serde_json::json!("low")));

    set_model_property(&mut model, "vendor_mode", serde_json::json!("strict"))
        .expect("valid property value");
    let mut values = Map::new();
    values.insert("vendor_mode".to_owned(), serde_json::json!("strict"));
    values.insert("provider_budget".to_owned(), serde_json::json!("4096"));
    set_model_properties(&mut model, values).expect("whole object validation and coercion");
    assert_eq!(model.provider_options["provider_budget"], 4096);
    assert_eq!(
        model.validated_provider_options().unwrap()["provider_budget"],
        4096
    );
}

/// Preserves the provider-owned schema from model configuration into runtime.
#[test]
fn configured_provider_schema_reaches_runtime_property_api() {
    let source = serde_json::json!({
        "providers": {
            "pluggy": {
                "api": "openai-completions",
                "models": [{
                    "id": "model-a",
                    "providerOptionsSchema": {
                        "version": 1,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "vendor_mode": {"type": "string", "enum": ["fast", "safe"]}
                            },
                            "additionalProperties": false
                        }
                    },
                    "vendor_mode": "fast"
                }]
            }
        }
    });
    let config: ModelsConfig = serde_json::from_value(source).expect("models config");
    let provider = config.providers.get("pluggy").expect("provider config");
    let model = provider_to_models("pluggy", provider, &EmptyConfigResolver)
        .expect("provider models")
        .remove(0);

    assert_eq!(model_property_descriptors(&model).unwrap().len(), 1);
    assert_eq!(
        model.validated_provider_options().unwrap()["vendor_mode"],
        "fast"
    );
}

/// Failed whole-object replacement leaves the prior configured map intact.
#[test]
fn invalid_model_property_replacement_is_transactional() {
    let mut model = Model::new(
        "model-a",
        "Model A",
        Api::OpenaiCompletions,
        "pluggy",
        "https://example.test",
    );
    model.provider_options_schema = Some(ProviderOptionsSchema {
        version: 1,
        schema: serde_json::json!({
            "type": "object",
            "properties": {"vendor_mode": {"type": "string"}},
            "required": ["vendor_mode"],
            "additionalProperties": false
        }),
    });
    model
        .provider_options
        .insert("vendor_mode".to_owned(), serde_json::json!("strict"));
    let before = model.provider_options.clone();

    assert!(set_model_properties(&mut model, Map::new()).is_err());
    assert_eq!(model.provider_options, before);
}
