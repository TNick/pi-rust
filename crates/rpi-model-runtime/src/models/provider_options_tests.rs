//! Regression fixtures for preserving opaque provider options as runtime data.

use super::*;

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
