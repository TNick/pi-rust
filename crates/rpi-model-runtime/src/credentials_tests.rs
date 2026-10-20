//! Tests protected credential precedence and host policy failures.

use std::sync::Mutex;

use super::{
    CredentialRequestSource, CredentialResolutionError, CredentialSource, HostCredentialMaterial,
    ProtectedCredentialResolver, resolve_credential_protected,
    resolve_credential_protected_with_policy,
};

/// Selects the test resolver's response to its first source request.
#[derive(Clone, Copy)]
enum Action {
    /// Report that the requested source is absent.
    Missing,
    /// Report that host policy denied the requested source.
    Denied,
}

/// Records source precedence calls and returns a configured test outcome.
struct TestResolver {
    /// Source descriptors received from the resolver function.
    requests: Mutex<Vec<CredentialRequestSource>>,
    /// Action returned for the first source request.
    first_action: Action,
}

impl ProtectedCredentialResolver<&'static str> for TestResolver {
    /// Implements the legacy miss-only resolver path.
    fn resolve(
        &self,
        _source: CredentialRequestSource,
    ) -> Option<HostCredentialMaterial<&'static str>> {
        None
    }

    /// Records source requests and returns the selected policy outcome.
    fn resolve_with_policy(
        &self,
        source: CredentialRequestSource,
    ) -> Result<Option<HostCredentialMaterial<&'static str>>, CredentialResolutionError> {
        self.requests.lock().unwrap().push(source.clone());
        match (&source, self.first_action) {
            (CredentialRequestSource::CliOverride { .. }, Action::Denied) => {
                Err(CredentialResolutionError::AccessDenied)
            }
            (CredentialRequestSource::CliOverride { .. }, Action::Missing) => Ok(None),
            (CredentialRequestSource::Stored { .. }, _) => Ok(Some(HostCredentialMaterial {
                material: "stored-secret",
                source: CredentialSource::StoredApiKey,
            })),
            _ => Ok(None),
        }
    }
}

/// Stops source precedence at a host authorization denial.
#[test]
fn denied_source_aborts_before_lower_priority_sources() {
    let resolver = TestResolver {
        requests: Mutex::new(Vec::new()),
        first_action: Action::Denied,
    };

    let error = resolve_credential_protected_with_policy("custom", None, None, &resolver)
        .expect_err("denial must abort credential selection");

    assert_eq!(error, CredentialResolutionError::AccessDenied);
    assert_eq!(
        resolver.requests.lock().unwrap().as_slice(),
        &[CredentialRequestSource::CliOverride {
            provider: "custom".to_owned()
        }]
    );
}

/// Maps policy denial to a safe missing result through the legacy API.
#[test]
fn legacy_resolver_api_maps_denial_to_missing_without_fallback() {
    let resolver = TestResolver {
        requests: Mutex::new(Vec::new()),
        first_action: Action::Denied,
    };

    let credential = resolve_credential_protected("custom", None, None, &resolver);

    assert_eq!(credential.material, None);
    assert_eq!(credential.source, CredentialSource::Missing);
    assert_eq!(
        resolver.requests.lock().unwrap().as_slice(),
        &[CredentialRequestSource::CliOverride {
            provider: "custom".to_owned()
        }]
    );
}

/// Allows a missing high-priority source to fall through to stored material.
#[test]
fn missing_source_falls_through_and_returns_selected_material() {
    let resolver = TestResolver {
        requests: Mutex::new(Vec::new()),
        first_action: Action::Missing,
    };

    let credential = resolve_credential_protected_with_policy("custom", None, None, &resolver)
        .expect("missing source should continue precedence");

    assert_eq!(credential.material, Some("stored-secret"));
    assert_eq!(credential.source, CredentialSource::StoredApiKey);
    assert_eq!(
        resolver.requests.lock().unwrap().as_slice(),
        &[
            CredentialRequestSource::CliOverride {
                provider: "custom".to_owned()
            },
            CredentialRequestSource::Stored {
                provider: "custom".to_owned()
            }
        ]
    );
}
