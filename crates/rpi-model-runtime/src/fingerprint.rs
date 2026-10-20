//! Safe fingerprints for the known output-affecting fields of model metadata.

use rpi_ai::Model;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Describes the coverage and safety limits of a model fingerprint.
///
/// # Public members
/// - `sha256`: Hexadecimal SHA-256 digest of canonicalized known model fields.
/// - `headers_excluded`: Always true; resolved request headers are omitted.
/// - `provider_options_included`: Always false; runtime-only provider options
///   are not included in this fingerprint.
///
/// # Private members
/// None.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelConfigurationFingerprint {
    /// SHA-256 digest of the canonical safe model representation.
    pub sha256: String,
    /// Whether potentially secret resolved request headers were excluded.
    pub headers_excluded: bool,
    /// Whether runtime-only provider options were included in this digest.
    pub provider_options_included: bool,
}

/// Fingerprints known model configuration without including resolved headers.
///
/// The digest covers serialized fields currently represented by
/// `rpi_ai::Model`, except `headers`, which are removed wholesale because they
/// may contain credentials. This is not a complete output-affecting
/// fingerprint when non-secret request headers affect execution. Runtime-only
/// provider options are deliberately excluded, as are unknown source
/// properties that do not map to the runtime option bag. Callers must not
/// treat the digest as proof that two requests are equivalent when excluded
/// headers or provider options differ.
///
/// # Parameters
/// - `model`: Resolved upstream model descriptor to fingerprint.
///
/// # Returns
/// A safe fingerprint with explicit coverage limitations.
pub fn model_configuration_fingerprint(model: &Model) -> ModelConfigurationFingerprint {
    let mut value = serde_json::to_value(model).expect("Model serialization is infallible");
    if let Value::Object(object) = &mut value {
        object.remove("headers");
        canonicalize(object);
    }
    let bytes = serde_json::to_vec(&value).expect("JSON value serialization is infallible");
    let digest = Sha256::digest(bytes);
    ModelConfigurationFingerprint {
        sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
        headers_excluded: true,
        provider_options_included: false,
    }
}

/// Sorts object keys recursively to stabilize their serialized order.
fn canonicalize(object: &mut serde_json::Map<String, Value>) {
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
