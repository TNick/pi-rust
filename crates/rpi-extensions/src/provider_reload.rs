//! Coordinate provider adapters and catalog refresh across extension reloads.

use std::sync::Arc;

use rpi_ai::Provider;
use tokio::runtime::Handle;

use crate::ExtensionSession;

/// Build the provider set associated with a freshly loaded extension session.
///
/// Host-configured providers remain first. Legacy registrations without a
/// declared provider manifest are appended as adapters; manifest-backed
/// providers must be supplied by the host's credential-bound resolver.
///
/// # Parameters
/// - `providers`: Host-resolved provider instances to retain.
/// - `session`: Fresh extension session to inspect for legacy providers.
/// - `runtime`: Runtime handle used by plugin provider adapters.
///
/// # Returns
/// Host providers followed by non-duplicate legacy extension adapters.
pub fn providers_for_reload(
    providers: &[Arc<dyn Provider>],
    session: &ExtensionSession,
    runtime: Handle,
) -> Vec<Arc<dyn Provider>> {
    let mut models = providers.to_vec();
    for provider in crate::PluggableProvider::from_session(session, runtime) {
        if session
            .snapshot()
            .is_some_and(|snapshot| snapshot.provider_manifest(provider.id()).is_some())
        {
            continue;
        }
        if !models.iter().any(|known| known.id() == provider.id()) {
            models.push(provider);
        }
    }
    models
}

/// Whether either session has declared providers that require host resolution.
///
/// # Parameters
/// - `previous`: Extension session currently attached to the host.
/// - `replacement`: Freshly loaded extension session.
///
/// # Returns
/// `true` when either session contains a registered provider manifest.
pub fn requires_provider_catalog_refresh(
    previous: &ExtensionSession,
    replacement: &ExtensionSession,
) -> bool {
    has_declared_providers(previous) || has_declared_providers(replacement)
}

/// Whether reload removed the active manifest-backed provider registration.
///
/// A provider is considered retained when the replacement session has a
/// provider registration with the same identity, matching host reload
/// behavior. Provider catalog resolution and credentials remain host-owned.
///
/// # Parameters
/// - `active_provider`: Active provider identity from the host model.
/// - `previous`: Extension session currently attached to the host.
/// - `replacement`: Freshly loaded extension session.
///
/// # Returns
/// `true` when the active provider had a manifest before reload and has no
/// matching provider registration afterward.
pub fn active_provider_was_removed(
    active_provider: &str,
    previous: &ExtensionSession,
    replacement: &ExtensionSession,
) -> bool {
    let was_manifest_backed = previous
        .snapshot()
        .is_some_and(|snapshot| snapshot.provider_manifest(active_provider).is_some());
    let remains_registered = replacement.snapshot().is_some_and(|snapshot| {
        snapshot
            .providers()
            .iter()
            .any(|provider| provider.provider_id == active_provider)
    });
    was_manifest_backed && !remains_registered
}

/// Check whether a session has any registered provider manifests.
///
/// # Parameters
/// - `session`: Extension session to inspect.
///
/// # Returns
/// `true` when a registered provider identity has a manifest.
fn has_declared_providers(session: &ExtensionSession) -> bool {
    session.snapshot().is_some_and(|snapshot| {
        snapshot
            .providers()
            .iter()
            .any(|provider| snapshot.provider_manifest(&provider.provider_id).is_some())
    })
}
