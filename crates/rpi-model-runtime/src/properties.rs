//! Schema-backed editing for provider-specific model properties.

use std::collections::BTreeMap;

use rpi_ai::{validate_json_value, AiError, Model};
use serde_json::{Map, Value};

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
    let normalized = validate_json_value(value, property_schema)?;
    model.provider_options.insert(key.to_owned(), normalized);
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
