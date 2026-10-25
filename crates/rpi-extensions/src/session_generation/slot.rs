//! The host's single publication gate for extension-session generations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

use crate::ExtensionSession;

use super::generation::{GenerationCandidate, GenerationError, InFlightGuard, SessionGeneration};
use super::identity::{GenerationId, GenerationIdentity};

/// The live extension-session generation, plus the history a host needs to drain
/// the generation it replaced.
///
/// Cheap to clone: every clone addresses the same publication gate, which is how
/// a host shares it across the TUI, the reload path, and tool dispatch.
///
/// Public members:
/// - [`ExtensionGenerationSlot::new`] / [`ExtensionGenerationSlot::empty`]:
///   create a slot with the startup session, or with none yet.
/// - [`ExtensionGenerationSlot::current`] / [`ExtensionGenerationSlot::current_id`]
///   / [`ExtensionGenerationSlot::current_identity`]: read the live generation.
/// - [`ExtensionGenerationSlot::resolve`]: address a captured generation id.
/// - [`ExtensionGenerationSlot::publish`] / [`ExtensionGenerationSlot::publish_with`]:
///   replace the live generation, optionally committing derived rows first.
/// - [`ExtensionGenerationSlot::begin_call`]: enter a call on the live
///   generation.
/// - [`ExtensionGenerationSlot::retire`]: stop new dispatch without replacing.
#[derive(Clone)]
pub struct ExtensionGenerationSlot {
    inner: Arc<SlotInner>,
}

struct SlotInner {
    /// The published generation. Every read of it takes the same lock the
    /// publication holds, so no reader can observe a half-swapped state.
    current: RwLock<Arc<SessionGeneration>>,
    /// Next id to allocate. Monotonic and never reused; a rejected publication
    /// still consumes its id.
    next_id: AtomicU64,
}

impl ExtensionGenerationSlot {
    /// Publish `session` as the first generation.
    ///
    /// # Parameters
    /// - `session`: Session loaded for the current host session.
    ///
    /// # Returns
    /// A slot whose live generation is generation `1`.
    pub fn new(session: ExtensionSession) -> Self {
        Self {
            inner: Arc::new(SlotInner {
                current: RwLock::new(Arc::new(SessionGeneration::new(
                    GenerationId::new(1),
                    session,
                    Vec::new(),
                    None,
                ))),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// Start a slot with no loaded session, for a host whose first real session
    /// must be published through a catalog commit.
    ///
    /// # Returns
    /// A slot whose live generation is an empty session (generation `1`), so
    /// readers never see a missing current generation.
    pub fn empty() -> Self {
        Self::new(ExtensionSession::none())
    }

    /// The live generation.
    ///
    /// # Returns
    /// A lease on the generation published right now.
    pub fn current(&self) -> Arc<SessionGeneration> {
        Arc::clone(
            &self
                .inner
                .current
                .read()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    /// Reads the live generation while holding the publication gate.
    ///
    /// A host can snapshot catalog rows inside `read` to ensure that its
    /// extension generation and catalog revision come from one publication.
    /// The closure must not call back into this slot or block indefinitely.
    ///
    /// # Parameters
    /// - `read`: Read operation that uses the current generation.
    ///
    /// # Returns
    /// The value produced by `read` while publication is excluded.
    pub fn with_current<R>(&self, read: impl FnOnce(&Arc<SessionGeneration>) -> R) -> R {
        let current = self
            .inner
            .current
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        read(&current)
    }

    /// The live generation's id.
    ///
    /// # Returns
    /// Id of the generation readers resolve to.
    pub fn current_id(&self) -> GenerationId {
        self.current().id()
    }

    /// The live generation's secret-free identity.
    ///
    /// # Returns
    /// Identity of the live generation, including any catalog revision it was
    /// published with.
    pub fn current_identity(&self) -> GenerationIdentity {
        self.current().identity()
    }

    /// Resolve a captured generation id against the live generation.
    ///
    /// A host that stored a generation id with derived rows uses this before it
    /// serves them, so a read captured under a superseded generation fails
    /// instead of pairing old rows with the current session.
    ///
    /// # Parameters
    /// - `id`: Generation id the caller captured earlier.
    ///
    /// # Returns
    /// The generation lease when `id` is still the live, active generation.
    ///
    /// # Errors
    /// [`GenerationError::Stale`] when a later generation is live, and
    /// [`GenerationError::Retired`] when `id` is live but was retired.
    pub fn resolve(&self, id: GenerationId) -> Result<Arc<SessionGeneration>, GenerationError> {
        let current = self.current();
        if current.id() != id {
            return Err(GenerationError::Stale {
                generation: id,
                current: current.id(),
            });
        }
        if !current.is_active() {
            return Err(GenerationError::Retired { generation: id });
        }
        Ok(current)
    }

    /// Enter one call against the live generation.
    ///
    /// This is the host-side gate for dispatch paths that have no staleness check
    /// of their own — notably plugin tool execution. The slot takes the
    /// publication gate's read lock to pick the generation, so an entry either
    /// addresses the generation that is live at that instant or, when the host
    /// holds a lease on a superseded generation and calls
    /// [`SessionGeneration::begin_call`] directly, is refused by that
    /// generation's closed admission state.
    ///
    /// # Returns
    /// The in-flight guard for the live generation.
    ///
    /// # Errors
    /// [`GenerationError::Retired`] when the live generation was retired and no
    /// replacement has been published yet.
    pub fn begin_call(&self) -> Result<InFlightGuard, GenerationError> {
        self.current().begin_call()
    }

    /// Replace the live generation without a catalog commit.
    ///
    /// # Parameters
    /// - `candidate`: Staged replacement session.
    ///
    /// # Returns
    /// The published generation and the superseded one.
    ///
    /// # Errors
    /// Never fails: this is [`ExtensionGenerationSlot::publish_with`] with a
    /// commit hook that always accepts, so no host rejection is possible.
    pub fn publish(
        &self,
        candidate: GenerationCandidate,
    ) -> Result<PublishedGeneration, GenerationError> {
        self.publish_with(candidate, |_| Ok(None))
    }

    /// Replace the live generation, committing derived rows inside the swap.
    ///
    /// `commit` runs while the publication gate is held, after the new id is
    /// allocated and before the new generation becomes visible. A host therefore
    /// performs its model-catalog write here — returning the revision it
    /// committed — so no reader can pair the new catalog rows with the old
    /// session or the old catalog rows with the new session. `commit` must not
    /// call back into this slot (the gate is already held) or block indefinitely.
    ///
    /// The superseded generation is retired inside the same gate region, before
    /// the replacement becomes readable, so there is no point at which a
    /// generation the host has already replaced still admits a new `begin_call`
    /// or dispatches through a snapshot/bridge whose staleness was never
    /// published. That retirement's side effects — flipping the
    /// [`RegistrySnapshot`](crate::RegistrySnapshot) flag and
    /// [`ActionBridge::invalidate`](crate::ActionBridge::invalidate), which
    /// reaches the host's cancel hooks — also run under the gate, so they must
    /// not call back into this slot either. The superseded generation's artifacts
    /// and pinned resources stay alive until every lease on it is released.
    ///
    /// # Parameters
    /// - `candidate`: Staged replacement session and pinned resources.
    /// - `commit`: Host hook that publishes rows derived from the candidate. It
    ///   receives the new generation's identity and returns the catalog revision
    ///   it committed, or `None` when the candidate contributes no rows.
    ///
    /// # Returns
    /// The published generation and the superseded one, which the host keeps to
    /// watch [`SessionGeneration::in_flight`] drain.
    ///
    /// # Errors
    /// [`GenerationError::CommitRejected`] when `commit` fails. Nothing changes in
    /// that case: the live generation stays live and active, the candidate's
    /// artifacts are released unloaded, and only the allocated id is consumed.
    pub fn publish_with<F>(
        &self,
        candidate: GenerationCandidate,
        commit: F,
    ) -> Result<PublishedGeneration, GenerationError>
    where
        F: FnOnce(&GenerationIdentity) -> Result<Option<u64>, String>,
    {
        let mut current = self
            .inner
            .current
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let id = GenerationId::new(
            self.inner
                .next_id
                .fetch_add(1, Ordering::SeqCst)
                .saturating_add(1),
        );
        let identity = GenerationIdentity::for_session(id, None, candidate.session());
        let catalog_revision =
            commit(&identity).map_err(|reason| GenerationError::CommitRejected { reason })?;
        let (session, retained) = candidate.into_parts();
        let published = Arc::new(SessionGeneration::new(
            id,
            session,
            retained,
            catalog_revision,
        ));
        // The outgoing generation is closed before the replacement is installed,
        // and both happen inside this gate region. A reader cannot see the
        // replacement while the outgoing session still admits entry, and it
        // cannot see the closed outgoing generation either because it is parked
        // on the gate: closing entry and installing the replacement are one
        // indivisible publication. A call that won the entry CAS just before the
        // close stays counted in `previous`, which is what keeps its leases alive
        // while the host drains that generation.
        let previous = Arc::clone(&*current);
        previous.retire();
        *current = Arc::clone(&published);
        Ok(PublishedGeneration {
            current: published,
            previous,
        })
    }

    /// Retire the live generation without replacing it.
    ///
    /// Use it to fail closed before teardown: new dispatch is refused, in-flight
    /// calls keep their artifacts, and the slot keeps a current generation so
    /// readers never see a hole.
    ///
    /// # Returns
    /// A lease on the retired generation, so the host can wait for it to drain.
    /// Idempotent; a second call returns the same retired generation.
    pub fn retire(&self) -> Arc<SessionGeneration> {
        let current = self.current();
        current.retire();
        current
    }
}

/// The result of one publication.
///
/// Public members:
/// - [`PublishedGeneration::current`]: the generation now live.
/// - [`PublishedGeneration::previous`]: the superseded generation, retired but
///   still holding its artifacts and pinned resources.
#[derive(Debug)]
pub struct PublishedGeneration {
    /// The generation this publication made live.
    pub current: Arc<SessionGeneration>,
    /// The generation this publication superseded.
    pub previous: Arc<SessionGeneration>,
}
