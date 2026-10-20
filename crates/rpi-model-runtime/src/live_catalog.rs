//! Shared live model catalog snapshots and update notifications.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use indexmap::IndexMap;
use rpi_ai::Model;
use tokio::sync::broadcast;

/// Publishes versioned model catalog changes to host products.
#[derive(Clone)]
pub struct ModelCatalog {
    /// Shared catalog state and notification channel.
    inner: Arc<ModelCatalogInner>,
}

/// Shared mutable model catalog state.
struct ModelCatalogInner {
    /// Provider-scoped models and their catalog revision.
    state: RwLock<ModelCatalogState>,
    /// Receives catalog update notifications.
    events: broadcast::Sender<ModelCatalogEvent>,
}

/// Mutable state protected by one lock for coherent snapshots.
#[derive(Default)]
struct ModelCatalogState {
    /// Last committed catalog revision.
    revision: u64,
    /// Models grouped by their host provider identity.
    providers: IndexMap<String, Vec<Model>>,
}

/// Immutable catalog view captured at one revision.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelCatalogSnapshot {
    /// Monotonic revision of the catalog state.
    pub revision: u64,
    /// Models flattened in provider insertion and per-provider order.
    pub models: Vec<Model>,
}

/// Notification emitted after one atomic model catalog update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelCatalogEvent {
    /// Revision available from a fresh snapshot.
    pub revision: u64,
    /// Providers whose rows or relative snapshot order changed in this update.
    pub provider_ids: Vec<String>,
}

impl ModelCatalog {
    /// Creates an empty catalog with revision zero.
    ///
    /// The catalog is cloneable; clones share state and notifications.
    ///
    /// # Returns
    /// A new empty model catalog.
    pub fn new() -> Self {
        let (events, _) = broadcast::channel(100);
        Self {
            inner: Arc::new(ModelCatalogInner {
                state: RwLock::new(ModelCatalogState::default()),
                events,
            }),
        }
    }

    /// Replaces all models associated with one provider.
    ///
    /// No event is emitted when the provider's model rows are unchanged.
    ///
    /// # Parameters
    /// - `provider_id`: Host-owned identity for the provider catalog.
    /// - `models`: Ordered model rows supplied by the host.
    ///
    /// # Returns
    /// The revision after applying the replacement.
    pub fn replace_provider_models(
        &self,
        provider_id: impl Into<String>,
        models: Vec<Model>,
    ) -> u64 {
        let provider_id = provider_id.into();
        let mut state = self
            .inner
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.providers.get(&provider_id) == Some(&models) {
            return state.revision;
        }
        state.providers.insert(provider_id.clone(), models);
        state.revision = state.revision.saturating_add(1);
        let revision = state.revision;
        let _ = self.inner.events.send(ModelCatalogEvent {
            revision,
            provider_ids: vec![provider_id],
        });
        revision
    }

    /// Atomically replaces the complete provider-to-model mapping.
    ///
    /// Providers absent from `providers` are removed. A changed mapping is
    /// committed with one revision and one event, so readers never observe an
    /// intermediate mixture of old and new provider catalogs.
    ///
    /// # Parameters
    /// - `providers`: Complete provider model mapping for the new catalog.
    ///
    /// # Returns
    /// The revision after applying the replacement.
    pub fn reconcile_providers(&self, providers: BTreeMap<String, Vec<Model>>) -> u64 {
        self.reconcile_ordered_providers(providers.into_iter().collect())
    }

    /// Atomically reconciles provider rows while preserving the supplied order.
    ///
    /// Providers absent from `providers` are removed. The mapping is committed
    /// as one revision and one notification.
    ///
    /// # Parameters
    /// - `providers`: Complete provider model mapping in desired snapshot order.
    ///
    /// # Returns
    /// The revision after applying the replacement.
    pub fn reconcile_ordered_providers(&self, providers: IndexMap<String, Vec<Model>>) -> u64 {
        let mut state = self
            .inner
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut provider_ids = state
            .providers
            .keys()
            .chain(providers.keys())
            .filter(|provider_id| state.providers.get(*provider_id) != providers.get(*provider_id))
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        if !state.providers.keys().eq(providers.keys()) {
            provider_ids.extend(state.providers.keys().chain(providers.keys()).cloned());
        }
        if provider_ids.is_empty() {
            return state.revision;
        }

        let provider_ids = provider_ids.into_iter().collect::<Vec<_>>();
        state.providers = providers;
        state.revision = state.revision.saturating_add(1);
        let revision = state.revision;
        let _ = self.inner.events.send(ModelCatalogEvent {
            revision,
            provider_ids,
        });
        revision
    }

    /// Atomically replaces an ordered provider slice and retains other rows.
    ///
    /// Rows outside the supplied slice remain present and precede the new
    /// slice. This lets a host component update its providers without deleting
    /// catalog rows owned by other components.
    ///
    /// # Parameters
    /// - `changes`: Ordered provider rows to replace, or `None` to remove that
    ///   provider.
    ///
    /// # Returns
    /// The revision after applying the changes.
    pub fn reconcile_provider_changes(&self, changes: IndexMap<String, Option<Vec<Model>>>) -> u64 {
        let mut state = self
            .inner
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut next = state
            .providers
            .iter()
            .filter(|(provider_id, _)| !changes.contains_key(*provider_id))
            .map(|(provider_id, models)| (provider_id.clone(), models.clone()))
            .collect::<IndexMap<_, _>>();
        for (provider_id, models) in changes {
            if let Some(models) = models {
                next.insert(provider_id, models);
            }
        }
        let mut changed_ids = state
            .providers
            .keys()
            .chain(next.keys())
            .filter(|provider_id| state.providers.get(*provider_id) != next.get(*provider_id))
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        if !state.providers.keys().eq(next.keys()) {
            changed_ids.extend(state.providers.keys().chain(next.keys()).cloned());
        }
        if changed_ids.is_empty() {
            return state.revision;
        }
        let changed_ids = changed_ids.into_iter().collect::<Vec<_>>();
        state.providers = next;
        state.revision = state.revision.saturating_add(1);
        let revision = state.revision;
        let _ = self.inner.events.send(ModelCatalogEvent {
            revision,
            provider_ids: changed_ids,
        });
        revision
    }

    /// Reconciles a flat provider/model list while preserving first-seen order.
    ///
    /// Models are grouped by their upstream provider identity. Provider order
    /// follows the first occurrence in `models`, and each provider's model
    /// order follows its occurrence order in that input.
    ///
    /// # Parameters
    /// - `models`: Complete ordered model rows for the new catalog.
    ///
    /// # Returns
    /// The revision after applying the replacement.
    pub fn reconcile_models(&self, models: Vec<Model>) -> u64 {
        let mut providers = IndexMap::<String, Vec<Model>>::new();
        for model in models {
            providers
                .entry(model.provider.clone())
                .or_default()
                .push(model);
        }
        self.reconcile_ordered_providers(providers)
    }

    /// Removes a provider's models from the live catalog.
    ///
    /// # Parameters
    /// - `provider_id`: Host-owned identity to remove.
    ///
    /// # Returns
    /// Whether a provider entry was removed.
    pub fn remove_provider(&self, provider_id: &str) -> bool {
        let mut state = self
            .inner
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.providers.shift_remove(provider_id).is_none() {
            return false;
        }
        state.revision = state.revision.saturating_add(1);
        let revision = state.revision;
        let _ = self.inner.events.send(ModelCatalogEvent {
            revision,
            provider_ids: vec![provider_id.to_owned()],
        });
        true
    }

    /// Captures all current provider models and the matching revision.
    ///
    /// # Returns
    /// A coherent immutable snapshot of the catalog.
    pub fn snapshot(&self) -> ModelCatalogSnapshot {
        let state = self
            .inner
            .state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ModelCatalogSnapshot {
            revision: state.revision,
            models: state
                .providers
                .values()
                .flat_map(|models| models.iter().cloned())
                .collect(),
        }
    }

    /// Captures the current models for one provider.
    ///
    /// # Parameters
    /// - `provider_id`: Host-owned identity to query.
    ///
    /// # Returns
    /// The provider's ordered model rows, or an empty list when absent.
    pub fn provider_models(&self, provider_id: &str) -> Vec<Model> {
        self.inner
            .state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .providers
            .get(provider_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Subscribes to provider catalog changes.
    ///
    /// # Returns
    /// A receiver notified after each provider-scoped or batch update.
    pub fn subscribe(&self) -> broadcast::Receiver<ModelCatalogEvent> {
        self.inner.events.subscribe()
    }
}

impl Default for ModelCatalog {
    /// Creates an empty shared model catalog.
    ///
    /// # Returns
    /// A new model catalog.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rpi_ai::{Api, Model};

    /// Builds one model row for live catalog tests.
    ///
    /// - `provider_id`: Provider identity.
    /// - `model_id`: Provider-native model identity.
    ///
    /// The corresponding upstream model.
    fn model(provider_id: &str, model_id: &str) -> Model {
        Model::new(
            model_id,
            model_id,
            Api::OpenaiCompletions,
            provider_id,
            "https://example.test",
        )
    }

    /// Verifies snapshots and subscribers observe provider-scoped updates.
    #[tokio::test]
    async fn updates_publish_coherent_snapshots_and_provider_events() {
        let catalog = ModelCatalog::new();
        let mut updates = catalog.subscribe();
        let first_revision =
            catalog.replace_provider_models("provider-a", vec![model("provider-a", "model-a")]);
        let first_event = updates.recv().await.expect("provider update event");
        let first_snapshot = catalog.snapshot();

        assert_eq!(first_revision, 1);
        assert_eq!(first_event.revision, first_snapshot.revision);
        assert_eq!(first_event.provider_ids, ["provider-a"]);
        assert_eq!(first_snapshot.models[0].id, "model-a");

        let second_revision =
            catalog.replace_provider_models("provider-a", vec![model("provider-a", "model-b")]);
        let second_event = updates.recv().await.expect("replacement event");
        let second_snapshot = catalog.snapshot();

        assert_eq!(second_revision, 2);
        assert_eq!(second_event.revision, second_snapshot.revision);
        assert_eq!(second_snapshot.models[0].id, "model-b");
        assert_eq!(catalog.provider_models("provider-a")[0].id, "model-b");
    }

    /// Verifies unchanged rows do not advance revisions or emit notifications.
    #[tokio::test]
    async fn unchanged_provider_rows_do_not_emit_updates() {
        let catalog = ModelCatalog::new();
        let models = vec![model("provider-a", "model-a")];
        catalog.replace_provider_models("provider-a", models.clone());
        let mut updates = catalog.subscribe();

        let revision = catalog.replace_provider_models("provider-a", models);

        assert_eq!(revision, 1);
        assert!(matches!(
            updates.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }

    /// Verifies full catalog reconciliation commits update and removal once.
    #[tokio::test]
    async fn reconciliation_updates_and_removes_providers_atomically() {
        let catalog = ModelCatalog::new();
        let mut initial = IndexMap::new();
        initial.insert("provider-a".to_owned(), vec![model("provider-a", "old-a")]);
        initial.insert("provider-b".to_owned(), vec![model("provider-b", "old-b")]);
        catalog.reconcile_ordered_providers(initial);
        let mut updates = catalog.subscribe();

        let mut refreshed = IndexMap::new();
        refreshed.insert("provider-a".to_owned(), vec![model("provider-a", "new-a")]);
        refreshed.insert("provider-c".to_owned(), vec![model("provider-c", "new-c")]);
        let revision = catalog.reconcile_ordered_providers(refreshed);
        let event = updates.recv().await.expect("one reconciliation event");
        let snapshot = catalog.snapshot();

        assert_eq!(revision, 2);
        assert_eq!(event.revision, revision);
        assert_eq!(
            event.provider_ids,
            ["provider-a", "provider-b", "provider-c"]
        );
        assert_eq!(
            snapshot
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["new-a", "new-c"]
        );
        assert!(catalog.provider_models("provider-b").is_empty());
        assert!(matches!(
            updates.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }

    /// Verifies reconciliation preserves changed provider display order.
    #[test]
    fn reconciliation_updates_provider_order() {
        let catalog = ModelCatalog::new();
        let mut initial = IndexMap::new();
        initial.insert("provider-b".to_owned(), vec![model("provider-b", "b")]);
        initial.insert("provider-a".to_owned(), vec![model("provider-a", "a")]);
        catalog.reconcile_ordered_providers(initial);
        let mut updates = catalog.subscribe();

        let mut reordered = IndexMap::new();
        reordered.insert("provider-a".to_owned(), vec![model("provider-a", "a")]);
        reordered.insert("provider-b".to_owned(), vec![model("provider-b", "b")]);
        let revision = catalog.reconcile_ordered_providers(reordered);
        let event = updates.try_recv().expect("provider order update event");

        assert_eq!(revision, 2);
        assert_eq!(event.revision, revision);
        assert_eq!(event.provider_ids, ["provider-a", "provider-b"]);
        assert_eq!(
            catalog
                .snapshot()
                .models
                .iter()
                .map(|model| model.provider.as_str())
                .collect::<Vec<_>>(),
            ["provider-a", "provider-b"]
        );
    }

    /// Keeps the original sorted-map caller shape source-compatible.
    #[test]
    fn reconciliation_accepts_legacy_btree_map_callers() {
        let catalog = ModelCatalog::new();
        let providers = std::collections::BTreeMap::from([
            ("provider-b".to_owned(), vec![model("provider-b", "b")]),
            ("provider-a".to_owned(), vec![model("provider-a", "a")]),
        ]);

        catalog.reconcile_providers(providers);

        assert_eq!(
            catalog
                .snapshot()
                .models
                .iter()
                .map(|model| model.provider.as_str())
                .collect::<Vec<_>>(),
            ["provider-a", "provider-b"]
        );
    }
}
