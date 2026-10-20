//! Provider credential types and host-injected authentication resolution.

use std::collections::BTreeMap;
use std::fmt;

use rpi_ai::Model;
use rpi_extensions::ProviderManifest;
use serde::{Deserialize, Serialize};

use crate::models::has_header_auth;

/// A persisted credential value associated with one provider identity.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Credential {
    /// A provider API key with an optional host-resolved environment overlay.
    ApiKey {
        /// The literal key or host-resolved configuration expression.
        key: Option<String>,
        /// Environment values scoped to this credential, when provided.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        env: Option<BTreeMap<String, String>>,
    },
    /// An OAuth token set associated with one provider identity.
    Oauth {
        /// Access token value, retained only in the host credential store.
        access: String,
        /// Refresh token value, retained only in the host credential store.
        #[serde(default)]
        refresh: String,
        /// Unix epoch expiration time, or zero when expiration is unknown.
        #[serde(default)]
        expires: i64,
    },
}

impl fmt::Debug for Credential {
    /// Formats credential metadata without revealing credential values.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey { env, .. } => f
                .debug_struct("ApiKey")
                .field("key", &"[redacted]")
                .field("env_names", &env.as_ref().map(BTreeMap::keys))
                .finish(),
            Self::Oauth { expires, .. } => f
                .debug_struct("Oauth")
                .field("access", &"[redacted]")
                .field("refresh", &"[redacted]")
                .field("expires", expires)
                .finish(),
        }
    }
}

/// Provider id to credential mapping supplied by the product host.
pub type AuthStore = BTreeMap<String, Credential>;

/// Non-secret source requested from a host credential resolver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialRequestSource {
    /// Resolve an explicit host-selected command-line credential override.
    CliOverride { provider: String },
    /// Resolve the provider's credential from host-owned persistent storage.
    Stored { provider: String },
    /// Resolve a configured provider API-key expression from host config.
    ModelsJson { provider: String },
    /// Resolve one named environment source through host policy.
    Environment { provider: String, name: String },
}

/// Secret material and safe source classification returned by the host.
pub struct HostCredentialMaterial<T> {
    /// Host-protected secret value.
    pub material: T,
    /// Safe source used to satisfy the request.
    pub source: CredentialSource,
}

impl<T> fmt::Debug for HostCredentialMaterial<T> {
    /// Formats credential source metadata without revealing host material.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostCredentialMaterial")
            .field("material", &"[redacted]")
            .field("source", &self.source)
            .finish()
    }
}

/// Reports protected credential resolution failures without secret details.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CredentialResolutionError {
    /// Host policy denied access to the requested source.
    #[error("credential source access denied")]
    AccessDenied,
    /// Host credential resolution could not complete safely.
    #[error("credential resolution failed")]
    ResolverFailure,
}

/// Resolves secret source requests without transferring plaintext to runtime.
pub trait ProtectedCredentialResolver<T>: Send + Sync {
    /// Resolves one non-secret source descriptor using host-owned facilities.
    ///
    /// For `ModelsJson`, the resolver must be bound to the exact immutable
    /// configuration snapshot used by the execution. The provider id selects
    /// its entry within that snapshot; callers must not redirect the request
    /// to a different current configuration.
    ///
    /// # Parameters
    /// - `source`: Provider and source kind, without credential values.
    ///
    /// # Returns
    /// Host-protected material and its safe source classification when found.
    fn resolve(&self, source: CredentialRequestSource) -> Option<HostCredentialMaterial<T>>;

    /// Resolves one source while distinguishing denial from a source miss.
    ///
    /// Implementations with authorization policy should override this method
    /// and return an error for denial or resolver failure. The default adapter
    /// preserves legacy behavior for existing implementations, which cannot
    /// report those errors.
    ///
    /// # Parameters
    /// - `source`: Provider and source kind, without credential values.
    ///
    /// # Returns
    /// `Ok(None)` when the source is absent, `Ok(Some(_))` when selected, or a
    /// non-secret error that stops source precedence.
    fn resolve_with_policy(
        &self,
        source: CredentialRequestSource,
    ) -> Result<Option<HostCredentialMaterial<T>>, CredentialResolutionError> {
        Ok(self.resolve(source))
    }
}

/// Protected request credential selected by upstream precedence.
pub struct ProtectedCredential<T> {
    /// Host-protected material for one provider request.
    pub material: Option<T>,
    /// Safe source classification for readiness projection.
    pub source: CredentialSource,
    /// Whether the request adapter should use bearer authorization.
    pub bearer: bool,
}

impl<T> fmt::Debug for ProtectedCredential<T> {
    /// Formats readiness metadata without revealing protected material.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProtectedCredential")
            .field("material", &self.material.as_ref().map(|_| "[redacted]"))
            .field("source", &self.source)
            .field("bearer", &self.bearer)
            .finish()
    }
}

/// Resolves a credential through upstream precedence using host-protected data.
///
/// The host receives only provider/source descriptors and resolves values from
/// its own config, credential store, or environment policy. The protected
/// material type is preserved without converting it to `String`.
///
/// # Parameters
/// - `provider`: Stable provider identity.
/// - `config`: Provider configuration from the same immutable snapshot used by
///   the resolver, used only for source presence and auth-header behavior.
/// - `manifest`: Optional provider package declaration.
/// - `resolver`: Host-owned protected-material resolver.
///
/// # Returns
/// The selected protected credential, or a safe missing/anonymous status. A
/// resolver error fails closed as `Missing` and does not continue precedence.
pub fn resolve_credential_protected<T>(
    provider: &str,
    config: Option<&crate::ProviderConfig>,
    manifest: Option<&ProviderManifest>,
    resolver: &dyn ProtectedCredentialResolver<T>,
) -> ProtectedCredential<T> {
    let bearer = config
        .and_then(|provider_config| provider_config.auth_header)
        .unwrap_or(false);
    match resolve_credential_protected_with_policy(provider, config, manifest, resolver) {
        Ok(credential) => credential,
        Err(_) => ProtectedCredential {
            material: None,
            source: CredentialSource::Missing,
            bearer,
        },
    }
}

/// Resolves a protected credential and aborts precedence on host policy errors.
///
/// A source miss falls through to the next configured source. Access denial or
/// resolver failure returns an error immediately, so lower-priority sources
/// are never queried and no credential material is returned to a provider
/// caller.
///
/// # Parameters
/// - `provider`: Stable provider identity.
/// - `config`: Provider configuration from the same immutable snapshot used by
///   the resolver, used only for source presence and auth-header behavior.
/// - `manifest`: Optional provider package declaration.
/// - `resolver`: Host-owned protected-material resolver.
///
/// # Returns
/// A protected credential only after a source is selected, or an anonymous or
/// missing status when every source misses.
///
/// # Errors
/// Returns a non-secret resolver error immediately when host policy denies a
/// source or resolution cannot complete safely.
pub fn resolve_credential_protected_with_policy<T>(
    provider: &str,
    config: Option<&crate::ProviderConfig>,
    manifest: Option<&ProviderManifest>,
    resolver: &dyn ProtectedCredentialResolver<T>,
) -> Result<ProtectedCredential<T>, CredentialResolutionError> {
    let bearer = config
        .and_then(|provider_config| provider_config.auth_header)
        .unwrap_or(false);
    let mut sources = vec![
        CredentialRequestSource::CliOverride {
            provider: provider.to_owned(),
        },
        CredentialRequestSource::Stored {
            provider: provider.to_owned(),
        },
    ];
    if config.is_some_and(|provider_config| provider_config.api_key.is_some()) {
        sources.push(CredentialRequestSource::ModelsJson {
            provider: provider.to_owned(),
        });
    }
    if manifest.is_none() && provider == "anthropic" {
        sources.push(CredentialRequestSource::Environment {
            provider: provider.to_owned(),
            name: "ANTHROPIC_AUTH_TOKEN".to_owned(),
        });
    }
    if let Some(name) = manifest
        .and_then(|value| value.api_key_env.as_deref())
        .or_else(|| {
            manifest
                .is_none()
                .then(|| crate::models::builtin_api_key_env(provider))
                .flatten()
        })
    {
        sources.push(CredentialRequestSource::Environment {
            provider: provider.to_owned(),
            name: name.to_owned(),
        });
    }
    for request_source in sources {
        if let Some(resolved) = resolver.resolve_with_policy(request_source)? {
            let is_oauth = resolved.source == CredentialSource::StoredOauth;
            let is_anthropic_auth_token = matches!(
                &resolved.source,
                CredentialSource::Environment(name) if name == "ANTHROPIC_AUTH_TOKEN"
            );
            return Ok(ProtectedCredential {
                material: Some(resolved.material),
                source: resolved.source,
                bearer: is_oauth || is_anthropic_auth_token || bearer,
            });
        }
    }
    let source = if manifest.is_some_and(|value| !value.requires_auth) {
        CredentialSource::Anonymous
    } else {
        CredentialSource::Missing
    };
    Ok(ProtectedCredential {
        material: None,
        source,
        bearer,
    })
}

/// Safe description of the source selected for a provider credential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialSource {
    /// Credential was supplied as an explicit host command-line override.
    CliOverride,
    /// Credential was loaded from the host's persistent auth store.
    StoredApiKey,
    /// Credential was loaded as an OAuth token.
    StoredOauth,
    /// Credential was configured with the provider model declarations.
    ModelsJson,
    /// Credential was resolved from a named environment variable.
    Environment(String),
    /// Credential was supplied as a model request header.
    Headers,
    /// Provider declaration allows requests without authentication.
    Anonymous,
    /// No usable credential source was found.
    Missing,
}

impl CredentialSource {
    /// Returns whether this source permits provider execution.
    pub fn is_present(&self) -> bool {
        !matches!(self, Self::Missing)
    }

    /// Returns a safe display value for the source.
    pub fn as_json_str(&self) -> &str {
        match self {
            Self::CliOverride => "cli",
            Self::StoredApiKey => "auth.json",
            Self::StoredOauth => "oauth",
            Self::ModelsJson => "models.json",
            Self::Environment(name) => name,
            Self::Headers => "headers",
            Self::Anonymous => "anonymous",
            Self::Missing => "none",
        }
    }
}

/// Credential resolution result used at the provider execution boundary.
///
/// The key is intentionally omitted from `Debug` and must not be serialized or
/// copied into catalog/session snapshots.
pub struct ResolvedCredential {
    /// Resolved secret material for constructing a provider request.
    pub key: Option<String>,
    /// Safe source classification for UI and provenance projections.
    pub source: CredentialSource,
    /// Whether a key is sent as a bearer authorization value.
    pub bearer: bool,
}

impl fmt::Debug for ResolvedCredential {
    /// Formats credential readiness without revealing secret material.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedCredential")
            .field("key", &self.key.as_ref().map(|_| "[redacted]"))
            .field("source", &self.source)
            .field("bearer", &self.bearer)
            .finish()
    }
}

/// Derives a safe provider readiness source from resolved credentials and a
/// credential-free or execution model catalog.
///
/// Header values are inspected only for presence. They are never returned or
/// copied into the resulting status.
///
/// # Parameters
/// - `provider`: Exact provider identity being checked.
/// - `credential_source`: Source returned by host-injected credential lookup.
/// - `models`: Model catalog that may contain provider-scoped auth headers.
///
/// # Returns
/// The credential source, `Headers` when a model supplies auth, or `Missing`.
pub fn readiness_from_catalog(
    provider: &str,
    credential_source: CredentialSource,
    models: &[Model],
) -> CredentialSource {
    if credential_source.is_present() {
        return credential_source;
    }
    if models
        .iter()
        .any(|model| model.provider == provider && has_header_auth(model.headers.as_ref()))
    {
        return CredentialSource::Headers;
    }
    CredentialSource::Missing
}

#[cfg(test)]
#[path = "credentials_tests.rs"]
mod tests;
