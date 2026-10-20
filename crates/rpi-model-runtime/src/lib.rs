//! Host-independent provider and model configuration for rpi products.
//!
//! Config files, credential storage, and environment access are supplied by
//! product hosts. Provider configuration, credential precedence, and model
//! conversion are shared by embedded and CLI products.

mod catalog;
mod credentials;
mod models;

pub use catalog::{apply_provider_defaults, build_catalog, resolve_provider_headers};
pub use credentials::{
    readiness_from_catalog, AuthStore, Credential, CredentialSource, ResolvedCredential,
};
pub use models::{
    builtin_api_key_env, has_header_auth, provider_to_models, resolve_credential,
    ConfigValueResolver, ModelDefinition, ModelsConfig, ProviderConfig,
};
