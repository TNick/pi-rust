//! Host-independent provider and model configuration for rpi products.
//!
//! Config files, credential storage, and environment access are supplied by
//! product hosts. Provider configuration, credential precedence, and model
//! conversion are shared by embedded and CLI products.

mod catalog;
mod config_value;
mod credential_store;
mod credentials;
mod fingerprint;
mod live_catalog;
mod models;
mod properties;
mod provider_factory;
mod provider_rows;
mod refresh;
mod safe_snapshot;
mod selection;

pub use catalog::{
    apply_provider_defaults, build_catalog, resolve_model_endpoints, resolve_provider_headers,
    ModelEndpointResolutionError,
};
pub use config_value::resolve_environment_template;
pub use credential_store::{
    delete_auth_credential, read_auth_store, upsert_auth_credential, write_auth_store,
    CredentialStoreError,
};
pub use credentials::{
    readiness_from_catalog, resolve_credential_protected, resolve_credential_protected_with_policy,
    resolve_host_selected_credential_with_policy, AuthStore, Credential, CredentialRequestSource,
    CredentialResolutionError, CredentialSource, HostCredentialMaterial, ProtectedCredential,
    ProtectedCredentialResolver, ResolvedCredential,
};
pub use fingerprint::{
    model_configuration_fingerprint, model_execution_fingerprint, ModelConfigurationFingerprint,
    ModelExecutionExclusion, ModelExecutionFingerprint,
};
pub use live_catalog::{ModelCatalog, ModelCatalogEvent, ModelCatalogSnapshot};
pub use models::{
    builtin_api_key_env, has_header_auth, provider_to_models, resolve_credential,
    ConfigValueResolver, ModelDefinition, ModelsConfig, ProviderConfig,
};
pub use properties::{
    model_property_descriptors, set_model_properties, set_model_property,
    update_model_config_properties, update_model_config_property, ModelPropertyConfigError,
    ModelPropertyDescriptor,
};
pub use provider_factory::{
    create_builtin_provider, create_builtin_provider_for_api, supports_builtin_provider_api,
    BuiltinProviderError,
};
pub use provider_rows::ProviderModelRows;
pub use provider_rows::{ModelsConfigFile, ModelsConfigFileError};
pub use safe_snapshot::{
    safe_model_execution_configuration, safe_model_snapshot, SafeModelCompatibility,
    SafeModelExecutionConfiguration, SafeModelSnapshot, SafeSamplingParameters,
};
pub use selection::{resolve_selection, ModelSelection, ResolvedSelection, SelectionError};
