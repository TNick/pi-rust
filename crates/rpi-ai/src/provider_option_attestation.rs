//! Builds secret-free attestations for provider-specific model options.

use crate::model::Model;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Describes how a provider handled one model-defined option.
///
/// # Public members
/// - `Applied`: The provider used the configured value unchanged.
/// - `Transformed`: The provider used a different effective value.
/// - `Ignored`: The provider did not use the configured option.
/// - `Unverifiable`: The provider's effective value was outside the safe schema.
///
/// # Private members
/// None.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderOptionDisposition {
    /// The provider applied the configured value unchanged.
    Applied,
    /// The provider applied a transformed value.
    Transformed,
    /// The provider ignored the configured option.
    Ignored,
    /// The effective value could not be represented safely.
    Unverifiable,
}

/// Records opaque identity for one schema-safe provider option.
///
/// The option name and values are replaced by domain-separated SHA-256
/// digests. This record is metadata only; it does not prove that a provider
/// behaved honestly.
///
/// # Public members
/// - `option_id_sha256`: Digest of the provider-defined option key.
/// - `disposition`: Whether the value was applied, transformed, ignored, or unverifiable.
/// - `effective_value_sha256`: Digest of the value used, if any.
///
/// # Private members
/// None.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderOptionAttestation {
    /// Digest of the provider-defined option key.
    pub option_id_sha256: String,
    /// Whether the configured value was applied, transformed, ignored, or unverifiable.
    pub disposition: ProviderOptionDisposition,
    /// Digest of the effective scalar value, absent for ignored or unverifiable options.
    pub effective_value_sha256: Option<String>,
}

/// Groups provider-option attestations and bounded omission metadata.
///
/// # Public members
/// - `options`: Attestations for schema-safe scalar options.
/// - `omitted_count`: Number of configured or applied options not represented.
///
/// # Private members
/// None.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderOptionsAttestation {
    /// Attestations for schema-safe scalar options.
    pub options: Vec<ProviderOptionAttestation>,
    /// Number of configured or applied options not represented.
    pub omitted_count: usize,
}

/// Attests which validated model options a provider applied or transformed.
///
/// Sensitive, composite, undeclared, and invalid options are omitted and only
/// contribute to `omitted_count`. The provider supplies the final option map
/// it actually used; an empty map marks every safe configured option ignored.
///
/// # Parameters
/// - `model`: Exact upstream model and option schema for the request.
/// - `effective_options`: Provider option values after provider preparation.
///
/// # Returns
/// Opaque per-option attestations and the number of omitted options.
pub fn attest_provider_options(
    model: &Model,
    effective_options: &BTreeMap<String, Value>,
) -> ProviderOptionsAttestation {
    let (configured, configured_omitted) = safe_provider_options(model);
    let Ok(Some(properties)) = model.provider_option_schema_properties() else {
        return ProviderOptionsAttestation {
            options: Vec::new(),
            omitted_count: configured_omitted,
        };
    };
    let mut attestation = ProviderOptionsAttestation {
        options: Vec::with_capacity(configured.len()),
        omitted_count: configured_omitted,
    };
    for (key, original) in &configured {
        let effective = effective_options.get(key);
        let schema = properties.get(key).expect("validated option has schema");
        let safe_effective = effective.filter(|value| safe_option(key, schema, value));
        let disposition = match (effective, safe_effective) {
            (Some(_), None) => ProviderOptionDisposition::Unverifiable,
            (Some(value), Some(_)) if value == original => ProviderOptionDisposition::Applied,
            (Some(_), Some(_)) => ProviderOptionDisposition::Transformed,
            (None, _) => ProviderOptionDisposition::Ignored,
        };
        attestation.options.push(ProviderOptionAttestation {
            option_id_sha256: digest("rpi-provider-option-key-v1", key.as_bytes()),
            disposition,
            effective_value_sha256: safe_effective
                .map(|value| canonical_value(value))
                .map(|bytes| digest("rpi-provider-option-value-v1", &bytes)),
        });
        if disposition == ProviderOptionDisposition::Unverifiable {
            attestation.omitted_count += 1;
        }
    }
    attestation.omitted_count += effective_options
        .keys()
        .filter(|key| !configured.contains_key(*key))
        .count();
    attestation
}

/// Validates option evidence against the exact upstream model schema.
///
/// This checks structural completeness and digest shape. It does not verify
/// the provider's claim about which effective value it used.
///
/// # Parameters
/// - `model`: Exact model descriptor bound to the provider request.
/// - `reported`: Provider-reported opaque option evidence.
///
/// # Returns
/// Whether every safe configured option is represented exactly once.
pub fn validate_provider_option_attestation(
    model: &Model,
    reported: &ProviderOptionsAttestation,
) -> bool {
    let (configured, configured_omitted) = safe_provider_options(model);
    if reported.omitted_count < configured_omitted
        || reported.omitted_count > 4096
        || reported.options.len() != configured.len()
    {
        return false;
    }
    let mut expected_ids = configured
        .keys()
        .map(|key| digest("rpi-provider-option-key-v1", key.as_bytes()))
        .collect::<std::collections::BTreeSet<_>>();
    let mut seen = std::collections::BTreeSet::new();
    for option in &reported.options {
        if !valid_sha256(&option.option_id_sha256)
            || !expected_ids.remove(&option.option_id_sha256)
            || !seen.insert(option.option_id_sha256.as_str())
        {
            return false;
        }
        let valid_effective_value = match option.disposition {
            ProviderOptionDisposition::Ignored | ProviderOptionDisposition::Unverifiable => {
                option.effective_value_sha256.is_none()
            }
            ProviderOptionDisposition::Applied | ProviderOptionDisposition::Transformed => option
                .effective_value_sha256
                .as_deref()
                .is_some_and(valid_sha256),
        };
        if !valid_effective_value {
            return false;
        }
    }
    expected_ids.is_empty()
}

/// Returns schema-safe scalar options and counts every option omitted.
///
/// # Parameters
/// - `model`: Model descriptor to validate and classify.
///
/// # Returns
/// A map of public scalar values and the count of unrepresented values.
pub fn safe_provider_options(model: &Model) -> (BTreeMap<String, Value>, usize) {
    let Ok(validated) = model.validated_provider_options() else {
        return (BTreeMap::new(), model.provider_options.len());
    };
    let Ok(Some(properties)) = model.provider_option_schema_properties() else {
        return (BTreeMap::new(), model.provider_options.len());
    };
    let mut safe = BTreeMap::new();
    for (key, value) in validated {
        let Some(schema) = properties.get(&key) else {
            continue;
        };
        if safe_option(key.as_str(), schema, &value) {
            safe.insert(key, value);
        }
    }
    let omitted = model.provider_options.len().saturating_sub(safe.len());
    (safe, omitted)
}

/// Checks a key/schema/value triple for safe scalar representation.
///
/// # Parameters
/// - `key`: Provider-defined option key.
/// - `schema`: Validated schema for the option.
/// - `value`: Validated option value.
///
/// # Returns
/// `true` when the option contains no secret marker and is a safe scalar.
fn safe_option(key: &str, schema: &Value, value: &Value) -> bool {
    let normalized = key.to_ascii_lowercase();
    let sensitive_name = [
        "secret",
        "password",
        "token",
        "credential",
        "api_key",
        "apikey",
        "authorization",
    ]
    .iter()
    .any(|marker| normalized.contains(marker));
    if sensitive_name || schema_is_sensitive(schema) {
        return false;
    }
    match (schema.get("type").and_then(Value::as_str), value) {
        (Some("boolean"), Value::Bool(_)) => true,
        (Some("integer"), Value::Number(number)) => {
            number.as_i64().is_some() || number.as_u64().is_some()
        }
        (Some("number"), Value::Number(number)) => number.as_f64().is_some_and(f64::is_finite),
        (Some("string"), Value::String(text)) => schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.contains(&Value::String(text.clone()))),
        _ => false,
    }
}

/// Detects secret, read-only, and composite schema markers recursively.
///
/// # Parameters
/// - `schema`: JSON Schema value to inspect.
///
/// # Returns
/// `true` when the schema is unsafe to include in a durable attestation.
fn schema_is_sensitive(schema: &Value) -> bool {
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
                properties
                    .iter()
                    .any(|(key, child)| safe_name_is_sensitive(key) || schema_is_sensitive(child))
            })
        || ["items", "allOf", "anyOf", "oneOf"]
            .iter()
            .filter_map(|key| schema.get(key))
            .any(|nested| match nested {
                Value::Array(items) => items.iter().any(schema_is_sensitive),
                value => schema_is_sensitive(value),
            })
        || matches!(
            schema.get("type").and_then(Value::as_str),
            Some("object" | "array")
        )
}

/// Checks whether a provider-defined name resembles a secret field.
///
/// # Parameters
/// - `name`: Schema property name.
///
/// # Returns
/// `true` when the name contains a common credential marker.
fn safe_name_is_sensitive(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
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
    .any(|marker| normalized.contains(marker))
}

/// Serializes an option value with recursively sorted object keys.
///
/// # Parameters
/// - `value`: JSON value to encode canonically.
///
/// # Returns
/// Canonical JSON bytes for the scalar value.
fn canonical_value(value: &Value) -> Vec<u8> {
    let mut value = value.clone();
    sort_value(&mut value);
    serde_json::to_vec(&value).expect("JSON value serialization is infallible")
}

/// Sorts all nested object keys before canonical encoding.
///
/// # Parameters
/// - `value`: JSON value to normalize in place.
fn sort_value(value: &mut Value) {
    match value {
        Value::Array(items) => items.iter_mut().for_each(sort_value),
        Value::Object(object) => {
            let sorted = object
                .iter_mut()
                .map(|(key, value)| {
                    sort_value(value);
                    (key.clone(), value.clone())
                })
                .collect::<BTreeMap<_, _>>();
            *object = Map::from_iter(sorted);
        }
        _ => {}
    }
}

/// Hashes a value with a versioned domain separator.
///
/// # Parameters
/// - `domain`: Stable hash purpose identifier.
/// - `value`: Bytes to hash.
///
/// # Returns
/// Lowercase hexadecimal SHA-256 digest.
fn digest(domain: &str, value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(value);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Checks whether text is a lowercase SHA-256 hexadecimal digest.
///
/// # Parameters
/// - `value`: Digest text to validate.
///
/// # Returns
/// Whether the text contains exactly 64 lowercase hexadecimal digits.
fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
