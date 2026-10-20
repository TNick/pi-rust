//! Host-independent provider and model configuration for rpi products.
//!
//! Config files, credential storage, and environment access are supplied by
//! product hosts. Provider configuration, credential precedence, and model
//! conversion are shared by embedded and CLI products.

mod credentials;
mod models;

pub use credentials::{
    readiness_from_catalog, AuthStore, Credential, CredentialSource, ResolvedCredential,
};
pub use models::{
    apply_provider_defaults, build_catalog, builtin_api_key_env, has_header_auth,
    provider_to_models, resolve_credential, resolve_provider_headers, ConfigValueResolver,
    ModelDefinition, ModelsConfig, ProviderConfig,
};
