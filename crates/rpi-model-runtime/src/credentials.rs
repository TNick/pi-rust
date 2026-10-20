//! Provider credential types and host-injected authentication resolution.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

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

/// Safe description of the source selected for a provider credential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialSource {
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
