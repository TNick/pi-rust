//! Host-independent provider and model configuration for rpi products.
//!
//! Config files, credential storage, and environment access are supplied by
//! product hosts. Provider configuration, credential precedence, and model
//! conversion are shared by embedded and CLI products.

mod catalog;
mod credentials;
mod fingerprint;
mod models;
mod safe_snapshot;
mod selection;

pub use catalog::{apply_provider_defaults, build_catalog, resolve_provider_headers};
pub use credentials::{
    AuthStore, Credential, CredentialRequestSource, CredentialResolutionError, CredentialSource,
    HostCredentialMaterial, ProtectedCredential, ProtectedCredentialResolver, ResolvedCredential,
    readiness_from_catalog, resolve_credential_protected, resolve_credential_protected_with_policy,
};
pub use fingerprint::{
    ModelConfigurationFingerprint, ModelExecutionExclusion, ModelExecutionFingerprint,
    model_configuration_fingerprint, model_execution_fingerprint,
};
pub use models::{
    ConfigValueResolver, ModelDefinition, ModelsConfig, ProviderConfig, builtin_api_key_env,
    has_header_auth, provider_to_models, resolve_credential,
};
pub use safe_snapshot::{
    SafeModelCompatibility, SafeModelExecutionConfiguration, SafeModelSnapshot,
    SafeSamplingParameters, safe_model_execution_configuration, safe_model_snapshot,
};
pub use selection::{ModelSelection, ResolvedSelection, SelectionError, resolve_selection};
