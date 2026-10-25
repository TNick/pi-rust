//! One published generation, its staged replacement, and the in-flight lease.

use std::any::Any;
use std::fmt;
use std::ops::Deref;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::{ExtensionSession, PluginKeepalive, RegistrySnapshot};

use super::admission::Admissions;
use super::identity::{GenerationId, GenerationIdentity};

/// Why an operation addressed at a generation was refused.
///
/// Variants name only generation ids and a host-authored reason, so an error
/// message can be logged or echoed without leaking plugin paths or credentials.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GenerationError {
    /// The addressed generation is no longer the live one.
    #[error("extension generation {generation} was superseded by generation {current}")]
    Stale {
        /// Generation the caller addressed.
        generation: GenerationId,
        /// Generation the slot publishes now.
        current: GenerationId,
    },
    /// The addressed generation was retired and admits no new calls.
    #[error("extension generation {generation} is retired and admits no new calls")]
    Retired {
        /// Retired generation the caller addressed.
        generation: GenerationId,
    },
    /// The host commit rejected the candidate before it was published, so the
    /// live generation is unchanged.
    #[error("extension generation candidate was rejected before publication: {reason}")]
    CommitRejected {
        /// Host-supplied reason for the rejection.
        reason: String,
    },
}

/// A staged replacement session, not yet published.
///
/// The host builds this after it has loaded the replacement (in-process through
/// [`load_session`](crate::load_session), or an isolated worker set it started
/// itself) and validated it against its own trust/root-authorization policy.
/// Publication is the point at which the candidate becomes observable, so a
/// candidate that is never published — or whose publication is rejected — simply
/// releases its artifacts.
///
/// Public members:
/// - [`GenerationCandidate::new`]: stage a loaded session.
/// - [`GenerationCandidate::retain`]: pin an extra host-owned resource to this
///   generation's lifetime.
/// - [`GenerationCandidate::session`]: inspect the staged session.
pub struct GenerationCandidate {
    session: ExtensionSession,
    /// Host-owned resources pinned to this generation. Type-erased because the
    /// replacement shape is the host's choice: an in-process host pins nothing,
    /// an isolated-worker host pins its worker client so the child process is
    /// not terminated while a call that used it is still running.
    retained: Vec<Arc<dyn Any + Send + Sync>>,
}

impl GenerationCandidate {
    /// Stage a loaded session as a replacement candidate.
    ///
    /// # Parameters
    /// - `session`: Session produced by the host's loader.
    ///
    /// # Returns
    /// The unpublished candidate.
    pub fn new(session: ExtensionSession) -> Self {
        Self {
            session,
            retained: Vec::new(),
        }
    }

    /// Pin one host-owned resource to the published generation's lifetime.
    ///
    /// The resource is dropped only when the last reference to the resulting
    /// generation is dropped — after every in-flight call released its guard —
    /// which is what makes an isolated worker set safe to retire.
    ///
    /// # Parameters
    /// - `resource`: Host handle to keep alive (for example an
    ///   `Arc<Mutex<IsolatedWorker>>`, whose drop terminates the child).
    ///
    /// # Returns
    /// The same candidate, with the resource pinned.
    pub fn retain(mut self, resource: Arc<dyn Any + Send + Sync>) -> Self {
        self.retained.push(resource);
        self
    }

    /// The staged session.
    ///
    /// # Returns
    /// The session this candidate would publish.
    pub fn session(&self) -> &ExtensionSession {
        &self.session
    }

    /// Consume the candidate into its parts for publication.
    ///
    /// # Returns
    /// The staged session and the resources pinned to it.
    pub(crate) fn into_parts(self) -> (ExtensionSession, Vec<Arc<dyn Any + Send + Sync>>) {
        (self.session, self.retained)
    }
}

/// One published extension-session generation.
///
/// A generation is the unit a host swaps, addresses, and drains. It owns the
/// loaded session (hence the plugin keepalive) and any resource the candidate
/// pinned, and it carries the staleness state that refuses new dispatch once it
/// is superseded. Hand out as `Arc<SessionGeneration>`; a lease is a cheap clone
/// that keeps the generation's resources mapped.
pub struct SessionGeneration {
    id: GenerationId,
    session: ExtensionSession,
    retained: Vec<Arc<dyn Any + Send + Sync>>,
    /// Closed plus admitted-call count in one word. Entry and supersession are
    /// decided by this single CAS, so a generation that a publication already
    /// replaced cannot admit a call, and a call that was admitted before the
    /// replacement stays counted until its guard is released.
    admissions: Admissions,
    catalog_revision: Option<u64>,
}

impl SessionGeneration {
    /// Build a published generation.
    ///
    /// # Parameters
    /// - `id`: Identity allocated by the publishing slot.
    /// - `session`: Loaded session this generation owns.
    /// - `retained`: Host resources pinned to the generation's lifetime.
    /// - `catalog_revision`: Catalog revision the host committed for it, if any.
    ///
    /// # Returns
    /// An active generation.
    pub(crate) fn new(
        id: GenerationId,
        session: ExtensionSession,
        retained: Vec<Arc<dyn Any + Send + Sync>>,
        catalog_revision: Option<u64>,
    ) -> Self {
        Self {
            id,
            session,
            retained,
            admissions: Admissions::open(),
            catalog_revision,
        }
    }

    /// This generation's identity.
    ///
    /// # Returns
    /// The id allocated at publication.
    pub fn id(&self) -> GenerationId {
        self.id
    }

    /// The catalog revision paired with this generation.
    ///
    /// # Returns
    /// The revision the host's commit hook returned at publication, or `None`
    /// when the generation was published without a catalog commit.
    pub fn catalog_revision(&self) -> Option<u64> {
        self.catalog_revision
    }

    /// The secret-free identity of this generation.
    ///
    /// # Returns
    /// Id, paired catalog revision, plugin count, tool names, and provider ids.
    pub fn identity(&self) -> GenerationIdentity {
        GenerationIdentity::for_session(self.id, self.catalog_revision, &self.session)
    }

    /// The loaded session behind this generation.
    ///
    /// # Returns
    /// The session this generation owns.
    pub fn session(&self) -> &ExtensionSession {
        &self.session
    }

    /// The merged registry snapshot of this generation, if any plugin loaded.
    ///
    /// # Returns
    /// The session's snapshot, or `None` when no plugin registered.
    pub fn snapshot(&self) -> Option<&RegistrySnapshot> {
        self.session.snapshot()
    }

    /// The plugin keepalive of this generation.
    ///
    /// # Returns
    /// A keepalive clone, usable as the sole owner of the mapped artifacts while
    /// this generation is referenced.
    pub fn keepalive(&self) -> Arc<PluginKeepalive> {
        self.session.keepalive()
    }

    /// Whether this generation still admits new calls.
    ///
    /// # Returns
    /// `false` once the generation was retired by a replacement or by
    /// [`ExtensionGenerationSlot::retire`](super::ExtensionGenerationSlot::retire).
    pub fn is_active(&self) -> bool {
        self.admissions.is_open()
    }

    /// Enter one call against this generation.
    ///
    /// The returned guard holds a lease on the generation, so the plugin
    /// artifacts and any pinned host resource stay valid for the whole call even
    /// if the generation is superseded while the call runs. Entry and retirement
    /// are one CAS on [`SessionGeneration::admissions`]: either this call is
    /// counted, or the generation is already closed and the call is refused.
    /// There is no intermediate state in which a closed generation reports a new
    /// in-flight call.
    ///
    /// # Returns
    /// The in-flight guard for this call.
    ///
    /// # Errors
    /// [`GenerationError::Retired`] when the generation no longer admits calls.
    pub fn begin_call(self: &Arc<Self>) -> Result<InFlightGuard, GenerationError> {
        if !self.admissions.admit() {
            return Err(GenerationError::Retired {
                generation: self.id,
            });
        }
        Ok(InFlightGuard {
            generation: Arc::clone(self),
        })
    }

    /// Calls currently admitted against this generation.
    ///
    /// A host uses this to decide when a superseded generation has drained — for
    /// example before it restarts an isolated worker or reports an unload.
    ///
    /// # Returns
    /// Number of live [`InFlightGuard`]s.
    pub fn in_flight(&self) -> usize {
        self.admissions.admitted()
    }

    /// Whether no call is admitted against this generation.
    ///
    /// # Returns
    /// `true` when [`SessionGeneration::in_flight`] is zero.
    pub fn is_idle(&self) -> bool {
        self.in_flight() == 0
    }

    /// Stop admitting new dispatch and mark the session's shared staleness
    /// state, without releasing any resource.
    ///
    /// Idempotent. The generation's keepalive and pinned resources stay alive as
    /// long as any lease references this generation, so callers already inside
    /// plugin code keep valid function pointers.
    pub(crate) fn retire(&self) {
        if !self.admissions.close() {
            return;
        }
        if let Some(snapshot) = self.session.snapshot_arc() {
            // The snapshot's flag is shared with the registry it was taken from,
            // so this is the same staleness state `assert_active` guards.
            snapshot.active_flag().store(false, Ordering::SeqCst);
        }
        if let Some(bridge) = self.session.action_bridge() {
            bridge.invalidate();
        }
    }
}

impl fmt::Debug for SessionGeneration {
    /// Formats only the secret-free generation state.
    ///
    /// # Parameters
    /// - `formatter`: Destination formatter.
    ///
    /// # Returns
    /// `Ok(())` after writing the state.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionGeneration")
            .field("id", &self.id)
            .field("catalog_revision", &self.catalog_revision)
            .field("active", &self.is_active())
            .field("in_flight", &self.in_flight())
            .field("loaded_plugins", &self.session.loaded_paths().len())
            .field("pinned_resources", &self.retained.len())
            .finish_non_exhaustive()
    }
}

/// One admitted call against a generation.
///
/// Held for the duration of a plugin tool call, provider request, or event
/// dispatch the host wants gated. Dropping it decrements the generation's
/// in-flight count; the generation itself — its mapped artifacts and pinned
/// resources — outlives the guard only while another lease exists, which is what
/// lets a host retire a generation without pulling the ground out from a running
/// call.
///
/// Public members:
/// - [`InFlightGuard::generation`]: the generation the call entered.
pub struct InFlightGuard {
    generation: Arc<SessionGeneration>,
}

impl InFlightGuard {
    /// The generation this call entered.
    ///
    /// # Returns
    /// The generation lease taken at entry.
    pub fn generation(&self) -> &Arc<SessionGeneration> {
        &self.generation
    }
}

impl Deref for InFlightGuard {
    type Target = SessionGeneration;

    fn deref(&self) -> &Self::Target {
        &self.generation
    }
}

impl fmt::Debug for InFlightGuard {
    /// Formats the guarded generation's secret-free state.
    ///
    /// # Parameters
    /// - `formatter`: Destination formatter.
    ///
    /// # Returns
    /// `Ok(())` after writing the state.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InFlightGuard")
            .finish_non_exhaustive()
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.generation.admissions.release();
    }
}
