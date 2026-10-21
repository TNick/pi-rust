//! Schema-backed editing for provider-specific model properties.

use std::collections::BTreeMap;

use rpi_ai::{validate_json_value, AiError, Api, Model};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::models::provider_api;
use crate::{ModelDefinition, ModelsConfig};

/// Errors from editing a schema-backed model property in saved configuration.
#[derive(Debug, Error)]
pub enum ModelPropertyConfigError {
    /// The provider is not configured.
    #[error("provider is not configured")]
    ProviderNotFound,
    /// The model is not configured under the provider.
    #[error("model is not configured")]
    ModelNotFound,
    /// The provider protocol is not supported by the generic property editor.
    #[error("provider API is unsupported")]
    UnsupportedProviderApi,
    /// No versioned property schema was declared by this provider/model.
    #[error("provider property schema is unavailable")]
    SchemaUnavailable,
    /// Upstream schema validation failed.
    #[error(transparent)]
    Validation(#[from] AiError),
}

/// One provider-declared property available to a host settings UI.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelPropertyDescriptor {
    /// Stable provider-defined property key.
    pub key: String,
    /// Display title from the JSON Schema, if provided.
    pub title: Option<String>,
    /// Help text from the JSON Schema, if provided.
    pub description: Option<String>,
    /// The property schema used to select an existing upstream UI control.
    pub schema: Value,
    /// Configured property value, if present.
    pub value: Option<Value>,
    /// Schema default, if present.
    pub default: Option<Value>,
    /// Whether the property is required by the enclosing object schema.
    pub required: bool,
}

impl ModelPropertyDescriptor {
    /// Reports whether a property is safe for a generic settings editor.
    ///
    /// Secret-like keys, write-only schemas, read-only schemas, and schemas
    /// with opaque composite values are excluded from generic editing.
    ///
    /// # Returns
    /// `true` when the property can be displayed and edited generically.
    pub fn is_safe_for_generic_editor(&self) -> bool {
        let key = self.key.to_ascii_lowercase();
        !key_is_sensitive(&key)
            && !schema_is_sensitive_or_read_only(&self.schema)
            && !matches!(
                self.schema.get("type").and_then(Value::as_str),
                Some("object" | "array")
            )
            && !self.value.as_ref().is_some_and(is_composite)
            && !self.default.as_ref().is_some_and(is_composite)
    }
}

/// Detects secret-like provider-defined property names.
fn key_is_sensitive(key: &str) -> bool {
    [
        "secret",
        "password",
        "token",
        "credential",
        "api_key",
        "apikey",
        "authorization",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

/// Detects sensitive or read-only markers throughout a property schema.
fn schema_is_sensitive_or_read_only(schema: &Value) -> bool {
    schema
        .get("writeOnly")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || schema
            .get("readOnly")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || schema
            .get("x-sensitive")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || schema
            .get("format")
            .and_then(Value::as_str)
            .is_some_and(|format| format.eq_ignore_ascii_case("password"))
        || schema
            .get("properties")
            .and_then(Value::as_object)
            .is_some_and(|properties| {
                properties.iter().any(|(key, child)| {
                    key_is_sensitive(&key.to_ascii_lowercase())
                        || schema_is_sensitive_or_read_only(child)
                })
            })
        || ["items", "allOf", "anyOf", "oneOf"]
            .iter()
            .filter_map(|key| schema.get(key))
            .any(|nested| match nested {
                Value::Array(items) => items.iter().any(schema_is_sensitive_or_read_only),
                value => schema_is_sensitive_or_read_only(value),
            })
}

/// Reports whether a configured value is composite.
fn is_composite(value: &Value) -> bool {
    value.is_object() || value.is_array()
}

/// Lists schema-backed settings without exposing unrecognized model fields.
pub fn model_property_descriptors(model: &Model) -> Result<Vec<ModelPropertyDescriptor>, AiError> {
    let Some(properties) = model.provider_option_schema_properties()? else {
        return Ok(Vec::new());
    };
    let schema = &model
        .provider_options_schema
        .as_ref()
        .expect("schema properties require schema")
        .schema;
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<std::collections::BTreeSet<_>>();

    Ok(properties
        .iter()
        .map(|(key, property_schema)| ModelPropertyDescriptor {
            key: key.clone(),
            title: property_schema
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_owned),
            description: property_schema
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned),
            schema: property_schema.clone(),
            value: model.provider_options.get(key).cloned(),
            default: property_schema.get("default").cloned(),
            required: required.contains(key.as_str()),
        })
        .collect())
}

/// Validates and stores one provider-declared property value.
///
/// Existing `rpi-tui::SettingsList` primitives can render scalar/enumerated
/// descriptors; free-form values can be edited with the upstream `Editor`.
/// The full object is revalidated before a provider request is built.
pub fn set_model_property(model: &mut Model, key: &str, value: Value) -> Result<(), AiError> {
    let properties = model
        .provider_option_schema_properties()?
        .ok_or_else(|| AiError::Schema {
            message: "provider property schema is unavailable".to_owned(),
        })?;
    let property_schema = properties.get(key).ok_or_else(|| AiError::Schema {
        message: "provider property is not declared by its schema".to_owned(),
    })?;
    validate_existing_properties(model, &properties)?;
    let normalized = validate_json_value(value, property_schema)?;
    model.provider_options.insert(key.to_owned(), normalized);
    Ok(())
}

/// Rejects invalid or undeclared pre-existing values before a single edit.
fn validate_existing_properties(
    model: &Model,
    properties: &Map<String, Value>,
) -> Result<(), AiError> {
    for (key, value) in &model.provider_options {
        let schema = properties.get(key).ok_or_else(|| AiError::Schema {
            message: "configured provider property is not declared by its schema".to_owned(),
        })?;
        validate_json_value(value.clone(), schema)?;
    }
    Ok(())
}

/// Replaces configured provider values after whole-object schema validation.
pub fn set_model_properties(model: &mut Model, values: Map<String, Value>) -> Result<(), AiError> {
    let mut candidate = model.clone();
    candidate.provider_options = values.into_iter().collect::<BTreeMap<_, _>>();
    candidate.provider_options = candidate
        .validated_provider_options()?
        .into_iter()
        .collect();
    model.provider_options = candidate.provider_options;
    Ok(())
}

/// Validates and saves one declared property to a `ModelsConfig` value.
///
/// This only mutates the serializable configuration after validation against
/// the provider-supplied schema. The host remains responsible for persisting
/// the updated config to its storage.
pub fn update_model_config_property(
    config: &mut ModelsConfig,
    provider_id: &str,
    model_id: &str,
    key: &str,
    value: Value,
) -> Result<(), ModelPropertyConfigError> {
    update_model_config(config, provider_id, model_id, |model| {
        set_model_property(model, key, value)
    })
}

/// Validates and replaces all declared property values in saved configuration.
pub fn update_model_config_properties(
    config: &mut ModelsConfig,
    provider_id: &str,
    model_id: &str,
    values: Map<String, Value>,
) -> Result<(), ModelPropertyConfigError> {
    update_model_config(config, provider_id, model_id, |model| {
        set_model_properties(model, values)
    })
}

/// Performs a transactional edit through the same runtime model validator.
fn update_model_config(
    config: &mut ModelsConfig,
    provider_id: &str,
    model_id: &str,
    edit: impl FnOnce(&mut Model) -> Result<(), AiError>,
) -> Result<(), ModelPropertyConfigError> {
    let provider = config
        .providers
        .get(provider_id)
        .ok_or(ModelPropertyConfigError::ProviderNotFound)?;
    let api = provider_api(provider.api.as_deref())
        .or_else(|| {
            provider
                .api
                .as_deref()
                .map(|api| Api::Other(api.to_owned()))
        })
        .ok_or(ModelPropertyConfigError::UnsupportedProviderApi)?;
    let definition = provider
        .models
        .iter()
        .find(|model| model.id == model_id)
        .ok_or(ModelPropertyConfigError::ModelNotFound)?;
    if definition.provider_options_schema.is_none() {
        return Err(ModelPropertyConfigError::SchemaUnavailable);
    }

    let mut runtime_model = runtime_model(provider_id, &api, definition);
    edit(&mut runtime_model)?;
    let normalized = runtime_model.provider_options;

    let definition = config
        .providers
        .get_mut(provider_id)
        .and_then(|provider| provider.models.iter_mut().find(|m| m.id == model_id))
        .ok_or(ModelPropertyConfigError::ModelNotFound)?;
    definition.provider_options = normalized;
    Ok(())
}

/// Builds the runtime view used by the shared schema-backed validator.
fn runtime_model(provider_id: &str, api: &Api, definition: &ModelDefinition) -> Model {
    let mut model = Model::new(
        definition.id.clone(),
        definition
            .name
            .clone()
            .unwrap_or_else(|| definition.id.clone()),
        api.clone(),
        provider_id,
        definition.base_url.clone().unwrap_or_default(),
    );
    model.provider_options = definition.provider_options.clone();
    model.provider_options_schema = definition.provider_options_schema.clone();
    model
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use crate::{ModelDefinition, ProviderConfig};
    use serde_json::json;

    fn config() -> ModelsConfig {
        let mut config = ModelsConfig::default();
        config.providers.insert(
            "extension".into(),
            ProviderConfig {
                name: None,
                base_url: Some("https://example.invalid".into()),
                api_key: None,
                api: Some("openai-completions".into()),
                headers: None,
                auth_header: None,
                models: vec![ModelDefinition {
                    id: "model".into(),
                    name: None,
                    base_url: None,
                    reasoning: None,
                    thinking_level_map: None,
                    context_window: None,
                    max_tokens: None,
                    cost: None,
                    sampling_params: None,
                    input: None,
                    headers: None,
                    compat: None,
                    provider_options_schema: Some(rpi_ai::ProviderOptionsSchema {
                        version: 1,
                        schema: json!({
                            "type": "object",
                            "properties": {
                                "region": {"type": "string", "enum": ["east", "west"]}
                            },
                            "additionalProperties": false
                        }),
                    }),
                    provider_options: BTreeMap::new(),
                }],
            },
        );
        config
    }

    #[test]
    fn config_property_update_is_schema_validated_and_persistable() {
        let mut config = config();
        update_model_config_property(&mut config, "extension", "model", "region", json!("west"))
            .unwrap();

        assert_eq!(
            config.providers["extension"].models[0].provider_options["region"],
            json!("west")
        );
        let serialized = serde_json::to_value(&config).unwrap();
        assert_eq!(
            serialized["providers"]["extension"]["models"][0]["region"],
            json!("west")
        );
    }

    #[test]
    fn invalid_config_property_does_not_mutate_saved_values() {
        let mut config = config();
        config.providers.get_mut("extension").unwrap().models[0]
            .provider_options
            .insert("region".into(), json!("east"));
        let result = update_model_config_property(
            &mut config,
            "extension",
            "model",
            "region",
            json!("north"),
        );

        assert!(result.is_err());
        assert_eq!(
            config.providers["extension"].models[0].provider_options["region"],
            json!("east")
        );
    }
}
