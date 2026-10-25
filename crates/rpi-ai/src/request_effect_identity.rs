//! Computes a safe identity for one prepared provider HTTP request.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use zeroize::{Zeroize, Zeroizing};

use crate::RequestSecret;

/// Computes a digest over the exact provider request without retaining its contents.
///
/// When the request carries authentication material, that material keys the
/// digest so a credential change changes request identity without exposing the
/// credential. Without authentication, the digest is emitted only when every
/// header and endpoint component is from a conservative public allowlist.
///
/// # Parameters
/// - `url`: Fully resolved provider request URL.
/// - `body`: Final JSON body after provider transformation.
/// - `headers`: Final provider headers before transport-added headers.
/// - `request_secret`: Optional host-protected credential and auth mode.
///
/// # Returns
/// A lowercase SHA-256 or HMAC-SHA-256 digest, or `None` when an unauthenticated
/// request contains an unclassified header or sensitive URL component.
pub(crate) fn fingerprint(
    url: &str,
    body: &serde_json::Value,
    headers: &BTreeMap<String, String>,
    request_secret: Option<(&RequestSecret, bool)>,
) -> Option<String> {
    let mut credential_material = Zeroizing::new(Vec::new());
    if let Some((secret, bearer)) = request_secret {
        credential_material.extend_from_slice(if bearer {
            b"authorization:bearer\0"
        } else {
            b"x-api-key:raw\0"
        });
        secret.with_secret(&mut |value| {
            credential_material.extend_from_slice(value.as_bytes());
        });
    }
    for (name, value) in headers {
        if sensitive_header(name) {
            credential_material.extend_from_slice(name.to_ascii_lowercase().as_bytes());
            credential_material.push(0);
            credential_material.extend_from_slice(value.as_bytes());
            credential_material.push(0);
        }
    }
    let keyed = !credential_material.is_empty();
    if !keyed
        && (url.contains('?')
            || url.contains('#')
            || url.split_once("://").is_some_and(|(_, rest)| {
                rest.split('/')
                    .next()
                    .is_some_and(|host| host.contains('@'))
            })
            || headers.keys().any(|name| !public_header(name)))
    {
        return None;
    }
    let mut canonical_headers = headers
        .iter()
        .map(|(name, value)| {
            (
                name.to_ascii_lowercase(),
                if sensitive_header(name) {
                    "<credential>".to_owned()
                } else {
                    value.clone()
                },
            )
        })
        .collect::<Vec<_>>();
    let serialized = serde_json::to_vec(&RequestEffectMaterial {
        method: "POST",
        url,
        headers: &canonical_headers,
        body,
    });
    for (_, value) in &mut canonical_headers {
        value.zeroize();
    }
    let material = Zeroizing::new(serialized.ok()?);
    let digest = if keyed {
        hmac_sha256(&credential_material, &material)
    } else {
        Sha256::digest(&material).into()
    };
    Some(hex(&digest))
}

/// Holds the canonical request fields included in an effect digest.
#[derive(Serialize)]
struct RequestEffectMaterial<'a> {
    /// HTTP method used by the provider adapter.
    method: &'static str,
    /// Fully resolved request URL.
    url: &'a str,
    /// Sorted request headers with credential values redacted.
    headers: &'a [(String, String)],
    /// Final provider request body.
    body: &'a serde_json::Value,
}

/// Returns whether a header value must be keyed and redacted.
///
/// # Parameters
/// - `name`: Header name to classify.
///
/// # Returns
/// `true` when the header can carry secret material.
fn sensitive_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization"
            | "proxy-authorization"
            | "x-api-key"
            | "api-key"
            | "cf-aig-authorization"
            | "cookie"
            | "set-cookie"
    )
}

/// Returns whether an unauthenticated header has a known public value class.
///
/// # Parameters
/// - `name`: Header name to classify.
///
/// # Returns
/// `true` when the header is safe to include in an unkeyed digest.
fn public_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "accept" | "content-type" | "anthropic-version" | "user-agent"
    )
}

/// Computes HMAC-SHA-256 using request credentials as an ephemeral key.
///
/// # Parameters
/// - `key_material`: Credential bytes used only for this digest operation.
/// - `message`: Canonical request bytes to authenticate.
///
/// # Returns
/// The 32-byte HMAC-SHA-256 result.
fn hmac_sha256(key_material: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK_SIZE: usize = 64;
    let mut normalized = Zeroizing::new([0_u8; BLOCK_SIZE]);
    if key_material.len() > BLOCK_SIZE {
        let digest = Sha256::digest(key_material);
        normalized[..digest.len()].copy_from_slice(&digest);
    } else {
        normalized[..key_material.len()].copy_from_slice(key_material);
    }
    let mut inner_pad = Zeroizing::new([0x36_u8; BLOCK_SIZE]);
    let mut outer_pad = Zeroizing::new([0x5c_u8; BLOCK_SIZE]);
    for index in 0..BLOCK_SIZE {
        inner_pad[index] ^= normalized[index];
        outer_pad[index] ^= normalized[index];
    }
    let mut inner = Sha256::new();
    inner.update(&inner_pad[..]);
    inner.update(message);
    let inner_digest = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(&outer_pad[..]);
    outer.update(inner_digest);
    let result = outer.finalize().into();
    normalized.zeroize();
    inner_pad.zeroize();
    outer_pad.zeroize();
    result
}

/// Converts one SHA-256 digest to lowercase hexadecimal.
///
/// # Parameters
/// - `digest`: Fixed-size digest bytes.
///
/// # Returns
/// The lowercase hexadecimal representation.
fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    //! Tests safe provider request identity behavior.

    use super::fingerprint;
    use crate::{RequestSecret, SecretMaterial};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Supplies test credentials through the protected request interface.
    struct TestSecret(String);

    impl SecretMaterial for TestSecret {
        /// Exposes the test credential for the duration of a callback.
        ///
        /// # Parameters
        /// - `consumer`: Callback that consumes the secret synchronously.
        fn with_secret(&self, consumer: &mut dyn FnMut(&str)) {
            consumer(&self.0);
        }
    }

    /// Builds a protected test credential handle.
    ///
    /// # Parameters
    /// - `value`: Credential content for the test case.
    ///
    /// # Returns
    /// An opaque request secret.
    fn secret(value: &str) -> RequestSecret {
        RequestSecret::new(Arc::new(TestSecret(value.to_owned())))
    }

    /// Returns a representative stable provider request body.
    ///
    /// # Returns
    /// A JSON body with model and prompt content.
    fn body() -> serde_json::Value {
        serde_json::json!({"model": "model-a", "input": "private prompt"})
    }

    /// Verifies equivalent authenticated requests have stable opaque identity.
    #[test]
    fn authenticated_identity_is_stable_and_credential_sensitive() {
        let request_secret = secret("credential-alpha");
        let headers = BTreeMap::from([("user-agent".to_owned(), "rpi-test".to_owned())]);
        let first = fingerprint(
            "https://provider.example/v1",
            &body(),
            &headers,
            Some((&request_secret, true)),
        );
        let repeated = fingerprint(
            "https://provider.example/v1",
            &body(),
            &headers,
            Some((&request_secret, true)),
        );
        let changed_credential = secret("credential-beta");
        let changed = fingerprint(
            "https://provider.example/v1",
            &body(),
            &headers,
            Some((&changed_credential, true)),
        );

        assert_eq!(first, repeated);
        assert_ne!(first, changed);
        let rendered = format!("{first:?} {repeated:?} {changed:?}");
        assert!(!rendered.contains("credential-alpha"));
        assert!(!rendered.contains("credential-beta"));
    }

    /// Verifies endpoint, body, and public-header changes affect identity.
    #[test]
    fn request_effect_fields_change_identity() {
        let headers = BTreeMap::from([("content-type".to_owned(), "application/json".to_owned())]);
        let original = fingerprint("https://provider.example/v1", &body(), &headers, None).unwrap();
        let changed_url =
            fingerprint("https://provider.example/v2", &body(), &headers, None).unwrap();
        let changed_body = fingerprint(
            "https://provider.example/v1",
            &serde_json::json!({"model": "model-b"}),
            &headers,
            None,
        )
        .unwrap();
        let changed_headers = fingerprint(
            "https://provider.example/v1",
            &body(),
            &BTreeMap::from([(
                "content-type".to_owned(),
                "application/problem+json".to_owned(),
            )]),
            None,
        )
        .unwrap();

        assert_ne!(original, changed_url);
        assert_ne!(original, changed_body);
        assert_ne!(original, changed_headers);
    }

    /// Verifies unclassified unauthenticated inputs cannot claim completeness.
    #[test]
    fn unsafe_unauthenticated_request_has_no_identity() {
        let headers = BTreeMap::from([("x-custom-token".to_owned(), "secret".to_owned())]);
        assert_eq!(
            fingerprint("https://provider.example/v1", &body(), &headers, None),
            None
        );
        let public_headers = BTreeMap::new();
        assert_eq!(
            fingerprint(
                "https://provider.example/v1?token=secret",
                &body(),
                &public_headers,
                None
            ),
            None
        );
    }
}
