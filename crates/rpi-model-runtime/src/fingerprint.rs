//! Safe fingerprints for the known output-affecting fields of model metadata.

use rpi_ai::Model;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Describes the coverage and safety limits of a model fingerprint.
///
/// # Public members
/// - `sha256`: Hexadecimal SHA-256 digest of canonicalized known model fields.
/// - `headers_excluded`: Always true; resolved request headers are omitted.
/// - `arbitrary_properties_preserved`: Always false; `rpi_ai::Model` has no
///   arbitrary property map.
///
/// # Private members
/// None.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelConfigurationFingerprint {
    /// SHA-256 digest of the canonical safe model representation.
    pub sha256: String,
    /// Whether potentially secret resolved request headers were excluded.
    pub headers_excluded: bool,
    /// Whether unknown properties outside `rpi_ai::Model` were captured.
    pub arbitrary_properties_preserved: bool,
}

/// Fingerprints known model configuration without including resolved headers.
///
/// The digest covers serialized fields currently represented by
/// `rpi_ai::Model`, except `headers`, which are removed wholesale because they
/// may contain credentials. This is not a complete output-affecting
/// fingerprint when non-secret request headers affect execution. The model
/// type has no arbitrary property map, so fields unknown to that type cannot
/// be retained or fingerprinted by this API. Callers must not treat the digest
/// as proof that two requests are equivalent when excluded headers or unknown
/// source properties differ.
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
        arbitrary_properties_preserved: false,
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

#[cfg(test)]
mod tests {
    //! Regression tests for safe model configuration fingerprints.

    use std::collections::BTreeMap;

    use rpi_ai::{Api, Model};

    use super::model_configuration_fingerprint;

    /// Constructs a small OpenAI-compatible model for fingerprint tests.
    fn model() -> Model {
        Model::new(
            "model",
            "Model",
            Api::OpenaiCompletions,
            "provider",
            "https://example.test",
        )
    }

    /// Verifies that resolved secret header values do not affect the digest.
    #[test]
    fn excludes_resolved_headers() {
        let mut first = model();
        first.headers = Some(BTreeMap::from([("Authorization".into(), "secret-a".into())]));
        let mut second = model();
        second.headers = Some(BTreeMap::from([("Authorization".into(), "secret-b".into())]));

        let first_fingerprint = model_configuration_fingerprint(&first);
        let second_fingerprint = model_configuration_fingerprint(&second);

        assert_eq!(first_fingerprint.sha256, second_fingerprint.sha256);
        assert!(first_fingerprint.headers_excluded);
        assert!(!first_fingerprint.arbitrary_properties_preserved);
    }

    /// Verifies that known request-shaping properties change the digest.
    #[test]
    fn includes_known_sampling_properties() {
        let mut first = model();
        first.sampling_params = Some(serde_json::Map::from_iter([(
            "temperature".into(),
            serde_json::json!(0.2),
        )]));
        let mut second = model();
        second.sampling_params = Some(serde_json::Map::from_iter([(
            "temperature".into(),
            serde_json::json!(0.8),
        )]));

        assert_ne!(
            model_configuration_fingerprint(&first).sha256,
            model_configuration_fingerprint(&second).sha256
        );
    }
}
