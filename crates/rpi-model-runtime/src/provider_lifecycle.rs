//! Secret-free provider authentication and credential lifecycle notifications.

pub use rpi_ai::ProviderAuthenticationOutcome;
use rpi_ai::ProviderAuthenticationResult;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, PoisonError, RwLock};

/// A secret-free provider authentication result tied to a catalog generation.
///
/// # Public members
/// - `provider_id`: Upstream provider identifier.
/// - `model_id`: Provider-scoped model identifier used for the check.
/// - `catalog_revision`: Catalog revision used for the request.
/// - `outcome`: Provider-reported authentication outcome.
///
/// # Private members
/// None.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderAuthenticationObservation {
    /// Upstream provider identifier.
    pub provider_id: String,
    /// Provider-scoped model identifier used for the check.
    pub model_id: String,
    /// Catalog revision used for the request.
    pub catalog_revision: u64,
    /// Provider-reported authentication outcome.
    pub outcome: ProviderAuthenticationOutcome,
}

/// A host-reported change to one provider's credential source.
///
/// The revision is an opaque monotonic host counter; no credential material is
/// accepted by this notification API.
///
/// # Public members
/// - `provider_id`: Upstream provider identifier.
/// - `credential_revision`: Host-assigned revision of the credential state.
///
/// # Private members
/// None.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderCredentialChange {
    /// Upstream provider identifier.
    pub provider_id: String,
    /// Host-assigned revision of the credential state.
    pub credential_revision: u64,
}

/// A provider authentication result or host credential-change notification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderLifecycleEvent {
    /// Provider runtime evidence about authentication.
    AuthenticationResult(ProviderAuthenticationObservation),
    /// Host notification that provider credential state changed.
    CredentialChanged(ProviderCredentialChange),
}

/// Broadcasts secret-free provider lifecycle events to runtime consumers.
///
/// Clones share one subscription list. Each subscriber receives subsequent
/// events independently; a disconnected subscriber is removed on publication.
#[derive(Clone, Default)]
pub struct ProviderLifecycleEvents {
    subscribers: Arc<RwLock<Vec<Sender<ProviderLifecycleEvent>>>>,
}

impl ProviderLifecycleEvents {
    /// Creates an event source with no subscribers.
    ///
    /// # Returns
    /// A new provider lifecycle event source.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribes to future provider lifecycle events.
    ///
    /// # Returns
    /// A receiver that yields each event published after this call.
    pub fn subscribe(&self) -> Receiver<ProviderLifecycleEvent> {
        let (sender, receiver) = mpsc::channel();
        self.subscribers
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .push(sender);
        receiver
    }

    /// Publishes a provider authentication result to every active subscriber.
    ///
    /// # Parameters
    /// - `result`: Secret-free result classified by the provider runtime.
    pub fn report_authentication(&self, result: ProviderAuthenticationObservation) {
        self.publish(ProviderLifecycleEvent::AuthenticationResult(result));
    }

    /// Publishes authentication evidence received from the upstream agent loop.
    ///
    /// # Parameters
    /// - `result`: Provider/model identity and outcome emitted for the request.
    /// - `catalog_revision`: Catalog revision used to resolve that model.
    pub fn report_agent_result(&self, result: ProviderAuthenticationResult, catalog_revision: u64) {
        self.report_authentication(ProviderAuthenticationObservation {
            provider_id: result.provider_id,
            model_id: result.model_id,
            catalog_revision,
            outcome: result.outcome,
        });
    }

    /// Notifies subscribers that host-owned provider credential state changed.
    ///
    /// # Parameters
    /// - `change`: Provider identity and host-assigned credential revision.
    pub fn notify_credential_changed(&self, change: ProviderCredentialChange) {
        self.publish(ProviderLifecycleEvent::CredentialChanged(change));
    }

    /// Sends one event to subscribers and removes disconnected receivers.
    ///
    /// # Parameters
    /// - `event`: Secret-free lifecycle event to publish.
    fn publish(&self, event: ProviderLifecycleEvent) {
        self.subscribers
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|subscriber| subscriber.send(event.clone()).is_ok());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ProviderAuthenticationObservation, ProviderAuthenticationOutcome, ProviderCredentialChange,
        ProviderLifecycleEvent, ProviderLifecycleEvents,
    };

    /// Delivers provider authentication evidence without exposing credentials.
    #[test]
    fn broadcasts_secret_free_authentication_results() {
        let events = ProviderLifecycleEvents::new();
        let receiver = events.subscribe();
        events.report_authentication(ProviderAuthenticationObservation {
            provider_id: "provider-a".to_owned(),
            model_id: "model-a".to_owned(),
            catalog_revision: 7,
            outcome: ProviderAuthenticationOutcome::Rejected,
        });
        assert_eq!(
            receiver.recv().unwrap(),
            ProviderLifecycleEvent::AuthenticationResult(ProviderAuthenticationObservation {
                provider_id: "provider-a".to_owned(),
                model_id: "model-a".to_owned(),
                catalog_revision: 7,
                outcome: ProviderAuthenticationOutcome::Rejected,
            })
        );
    }

    /// Converts an upstream agent result into generation-bound lifecycle evidence.
    #[test]
    fn agent_authentication_result_keeps_its_catalog_revision() {
        let events = ProviderLifecycleEvents::new();
        let receiver = events.subscribe();
        events.report_agent_result(
            rpi_ai::ProviderAuthenticationResult {
                provider_id: "provider-b".to_owned(),
                model_id: "model-b".to_owned(),
                outcome: ProviderAuthenticationOutcome::Accepted,
            },
            19,
        );

        assert_eq!(
            receiver.recv().unwrap(),
            ProviderLifecycleEvent::AuthenticationResult(ProviderAuthenticationObservation {
                provider_id: "provider-b".to_owned(),
                model_id: "model-b".to_owned(),
                catalog_revision: 19,
                outcome: ProviderAuthenticationOutcome::Accepted,
            })
        );
    }

    /// Delivers host credential changes and prunes disconnected receivers.
    #[test]
    fn broadcasts_credential_changes_and_removes_closed_subscribers() {
        let events = ProviderLifecycleEvents::new();
        let receiver = events.subscribe();
        let closed = events.subscribe();
        drop(closed);
        events.notify_credential_changed(ProviderCredentialChange {
            provider_id: "provider-a".to_owned(),
            credential_revision: 12,
        });
        assert_eq!(
            receiver.recv().unwrap(),
            ProviderLifecycleEvent::CredentialChanged(ProviderCredentialChange {
                provider_id: "provider-a".to_owned(),
                credential_revision: 12,
            })
        );
    }
}
