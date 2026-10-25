//! Atomic extension-session and provider-catalog generation publication.

use std::sync::Arc;

use rpi_ai::Model;
use rpi_extensions::{
    ExtensionGenerationSlot, GenerationCandidate, GenerationError, PublishedGeneration,
    SessionGeneration,
};

use crate::{ModelCatalog, ModelCatalogSnapshot};

/// Failure while staging or publishing an extension catalog generation.
#[derive(Debug)]
pub enum ExtensionCatalogRefreshError<E> {
    /// Candidate loading, policy checks, or catalog derivation failed.
    Stage(E),
    /// Atomic publication rejected the staged candidate.
    Publish(GenerationError),
}

/// Stages a complete extension generation and publishes it with its catalog.
///
/// The closure owns package discovery, trust/root authorization, loading, and
/// provider resolution. It must return a fully validated candidate and the
/// complete ordered model set. The closure runs before publication; on failure
/// the live session and catalog remain untouched. Successful staging delegates
/// to the shared generation gate, so paired readers observe one generation.
///
/// # Parameters
/// - `slot`: Shared extension generation slot.
/// - `catalog`: Shared provider/model catalog.
/// - `stage`: Host policy and staging operation that returns the candidate and
///   its complete catalog rows.
///
/// # Returns
/// The published generation and the retired previous generation.
///
/// # Errors
/// Returns `Stage` when the host staging closure fails, or `Publish` when the
/// shared publication gate rejects the candidate.
pub fn refresh_extension_catalog_generation<E, F>(
    slot: &ExtensionGenerationSlot,
    catalog: &ModelCatalog,
    stage: F,
) -> Result<PublishedGeneration, ExtensionCatalogRefreshError<E>>
where
    F: FnOnce() -> Result<(GenerationCandidate, Vec<Model>), E>,
{
    let (candidate, models) = stage().map_err(ExtensionCatalogRefreshError::Stage)?;
    publish_extension_catalog_generation(slot, catalog, candidate, models)
        .map_err(ExtensionCatalogRefreshError::Publish)
}

/// One coherent view of the live extension session and provider catalog.
///
/// Read this through [`snapshot_extension_catalog_generation`] so publication
/// cannot replace one member while the other is being captured.
///
/// # Public members
/// - `extension`: Lease on the live extension session generation.
/// - `catalog`: Model catalog snapshot published with that generation.
///
/// # Private members
/// None.
#[derive(Clone, Debug)]
pub struct ExtensionCatalogSnapshot {
    /// Lease on the live extension session generation.
    pub extension: Arc<SessionGeneration>,
    /// Provider catalog snapshot committed with the extension generation.
    pub catalog: ModelCatalogSnapshot,
}

/// Publishes a staged extension session with its complete provider catalog.
///
/// The slot keeps readers behind its publication gate while the catalog is
/// reconciled and the extension session is replaced. Hosts should capture
/// paired reads with [`snapshot_extension_catalog_generation`].
///
/// # Parameters
/// - `slot`: Shared upstream extension generation slot.
/// - `catalog`: Shared upstream provider/model catalog.
/// - `candidate`: Fully staged extension session and retained resources.
/// - `models`: Complete provider/model set derived for the candidate generation.
///
/// # Returns
/// The newly published extension generation and the retired previous generation.
///
/// # Errors
/// Returns a generation error if publication is rejected; the old generation
/// and catalog remain active in that case.
pub fn publish_extension_catalog_generation(
    slot: &ExtensionGenerationSlot,
    catalog: &ModelCatalog,
    candidate: GenerationCandidate,
    models: Vec<Model>,
) -> Result<PublishedGeneration, GenerationError> {
    slot.publish_with(candidate, |_| {
        Ok(Some(catalog.reconcile_models_snapshot(models).revision))
    })
}

/// Captures the live extension session and provider catalog under one gate.
///
/// # Parameters
/// - `slot`: Shared upstream extension generation slot.
/// - `catalog`: Shared upstream provider/model catalog.
///
/// # Returns
/// A session lease and catalog snapshot from the same published generation.
pub fn snapshot_extension_catalog_generation(
    slot: &ExtensionGenerationSlot,
    catalog: &ModelCatalog,
) -> ExtensionCatalogSnapshot {
    slot.with_current(|extension| ExtensionCatalogSnapshot {
        extension: Arc::clone(extension),
        catalog: catalog.snapshot(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use rpi_ai::{Api, Model};
    use rpi_extensions::{ExtensionGenerationSlot, ExtensionSession, GenerationCandidate};

    use crate::{
        publish_extension_catalog_generation, refresh_extension_catalog_generation,
        snapshot_extension_catalog_generation, ExtensionCatalogRefreshError, ModelCatalog,
    };

    /// Publishes model rows and their extension session as one readable generation.
    #[test]
    fn extension_and_catalog_snapshot_share_one_generation() {
        let slot = ExtensionGenerationSlot::empty();
        let catalog = ModelCatalog::new();
        let model = Model::new(
            "model-a",
            "Model A",
            Api::OpenaiCompletions,
            "provider-a",
            "https://example.test",
        );

        let published = publish_extension_catalog_generation(
            &slot,
            &catalog,
            GenerationCandidate::new(ExtensionSession::none()),
            vec![model.clone()],
        )
        .expect("publish complete provider generation");
        let snapshot = snapshot_extension_catalog_generation(&slot, &catalog);

        assert_eq!(snapshot.extension.id(), published.current.id());
        assert_eq!(
            snapshot.extension.catalog_revision(),
            Some(snapshot.catalog.revision)
        );
        assert_eq!(snapshot.catalog.models, vec![model]);
    }

    /// Blocks paired reads while the catalog is committed ahead of a session swap.
    #[test]
    fn paired_snapshot_cannot_observe_the_catalog_before_session_publication() {
        let slot = ExtensionGenerationSlot::empty();
        let catalog = ModelCatalog::new();
        let writer_slot = slot.clone();
        let writer_catalog = catalog.clone();
        let (catalog_committed, committed) = mpsc::channel();
        let (allow_publish, publish) = mpsc::channel();
        let writer = thread::spawn(move || {
            writer_slot
                .publish_with(GenerationCandidate::new(ExtensionSession::none()), |_| {
                    let revision = writer_catalog
                        .reconcile_models_snapshot(vec![Model::new(
                            "model-b",
                            "Model B",
                            Api::OpenaiCompletions,
                            "provider-b",
                            "https://example.test",
                        )])
                        .revision;
                    catalog_committed.send(()).unwrap();
                    publish.recv().unwrap();
                    Ok(Some(revision))
                })
                .expect("publish generation after catalog commit");
        });

        committed.recv().expect("catalog commit reached the gate");
        let reader_slot = slot.clone();
        let reader_catalog = catalog.clone();
        let (snapshot_ready, ready) = mpsc::channel();
        let reader = thread::spawn(move || {
            let snapshot = snapshot_extension_catalog_generation(&reader_slot, &reader_catalog);
            snapshot_ready.send(snapshot).unwrap();
        });

        assert!(ready.recv_timeout(Duration::from_millis(50)).is_err());
        allow_publish
            .send(())
            .expect("release the publication gate");
        let snapshot = ready
            .recv_timeout(Duration::from_secs(2))
            .expect("paired read resumes after publication");
        assert_eq!(
            snapshot.extension.catalog_revision(),
            Some(snapshot.catalog.revision)
        );
        assert_eq!(snapshot.catalog.models[0].provider, "provider-b");
        writer.join().expect("publication thread settles");
        reader.join().expect("snapshot reader settles");
    }

    /// Leaves the published session and catalog unchanged when staging fails.
    #[test]
    fn failed_refresh_keeps_the_previous_generation_and_catalog() {
        let slot = ExtensionGenerationSlot::empty();
        let catalog = ModelCatalog::new();
        let initial = Model::new(
            "model-a",
            "Model A",
            Api::OpenaiCompletions,
            "provider-a",
            "https://example.test",
        );
        publish_extension_catalog_generation(
            &slot,
            &catalog,
            GenerationCandidate::new(ExtensionSession::none()),
            vec![initial.clone()],
        )
        .expect("publish initial generation");
        let before = snapshot_extension_catalog_generation(&slot, &catalog);

        let result = refresh_extension_catalog_generation(&slot, &catalog, || {
            Err::<(GenerationCandidate, Vec<Model>), _>("package trust denied")
        });

        assert!(matches!(
            result,
            Err(ExtensionCatalogRefreshError::Stage(_))
        ));
        let after = snapshot_extension_catalog_generation(&slot, &catalog);
        assert_eq!(after.extension.id(), before.extension.id());
        assert_eq!(after.catalog.revision, before.catalog.revision);
        assert_eq!(after.catalog.models, vec![initial]);
    }

    /// Stages and publishes the replacement session and full model set together.
    #[test]
    fn refresh_publishes_staged_session_and_complete_catalog() {
        let slot = ExtensionGenerationSlot::empty();
        let catalog = ModelCatalog::new();
        let model = Model::new(
            "model-b",
            "Model B",
            Api::OpenaiCompletions,
            "provider-b",
            "https://example.test",
        );

        let published = refresh_extension_catalog_generation(&slot, &catalog, || {
            Ok::<_, &'static str>((
                GenerationCandidate::new(ExtensionSession::none()),
                vec![model.clone()],
            ))
        })
        .expect("publish staged provider package generation");
        let paired = snapshot_extension_catalog_generation(&slot, &catalog);

        assert_eq!(paired.extension.id(), published.current.id());
        assert_eq!(
            paired.extension.catalog_revision(),
            Some(paired.catalog.revision)
        );
        assert_eq!(paired.catalog.models, vec![model]);
    }
}
