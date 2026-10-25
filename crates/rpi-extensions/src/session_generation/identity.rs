//! Secret-free identity for one published extension generation.

use std::fmt;

use serde::Serialize;

use crate::ExtensionSession;

/// Monotonic identity of one published extension-session generation.
///
/// Ids are handed out by [`ExtensionGenerationSlot`](super::ExtensionGenerationSlot)
/// and are never reused, so a host can bind derived state (model-catalog rows,
/// readiness projections, a cached tool list) to the exact generation it read and
/// detect a superseded read. The value carries no plugin, credential, or
/// endpoint information and is safe to log or send to a client.
///
/// Public members:
/// - [`GenerationId::new`]: build an id from its numeric value.
/// - [`GenerationId::get`]: read the numeric value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct GenerationId(u64);

impl GenerationId {
    /// Build an id with the given value.
    ///
    /// # Parameters
    /// - `value`: Numeric id value. Slot publication allocates ids internally;
    ///   callers only need this to rebuild an id a host persisted.
    ///
    /// # Returns
    /// The id with that value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The numeric id value.
    ///
    /// # Returns
    /// The value this id was built from.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for GenerationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// The secret-free description of one generation.
///
/// A host uses this to label the rows it derives from a generation and to report
/// which generation a projection came from. It deliberately excludes every field
/// that could identify a plugin artifact or how a provider authenticates: plugin
/// file paths, [`ProviderManifest::api_key_env`](crate::ProviderManifest),
/// provider `base_url`/`api_style`, OAuth manifest JSON, and any credential
/// value. Logging or echoing this value does not disclose either.
///
/// Public members:
/// - [`GenerationIdentity::id`]: the generation this identity describes.
/// - [`GenerationIdentity::catalog_revision`]: host-committed catalog revision
///   paired with this generation, when publication ran a commit hook.
/// - [`GenerationIdentity::loaded_plugins`]: number of plugin artifacts loaded
///   into this generation.
/// - [`GenerationIdentity::tools`]: extension tool names, registration order.
/// - [`GenerationIdentity::providers`]: registered provider ids, registration
///   order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GenerationIdentity {
    /// The generation this identity describes.
    pub id: GenerationId,
    /// Catalog revision the host committed for this generation, or `None` when
    /// the publication carried no catalog commit.
    pub catalog_revision: Option<u64>,
    /// Number of plugin artifacts that loaded into this generation.
    pub loaded_plugins: usize,
    /// Extension tool names in registration order.
    pub tools: Vec<String>,
    /// Registered provider ids in registration order.
    pub providers: Vec<String>,
}

impl GenerationIdentity {
    /// Project the secret-free identity of `session` under `id`.
    ///
    /// # Parameters
    /// - `id`: Generation identity to attach.
    /// - `catalog_revision`: Host-committed catalog revision paired with `id`.
    /// - `session`: Session whose registrations are projected.
    ///
    /// # Returns
    /// The identity for that session.
    pub(crate) fn for_session(
        id: GenerationId,
        catalog_revision: Option<u64>,
        session: &ExtensionSession,
    ) -> Self {
        let snapshot = session.snapshot();
        Self {
            id,
            catalog_revision,
            loaded_plugins: session.loaded_paths().len(),
            tools: snapshot
                .map(|snapshot| {
                    snapshot
                        .tools()
                        .iter()
                        .map(|tool| tool.tool.name.clone())
                        .collect()
                })
                .unwrap_or_default(),
            providers: snapshot
                .map(|snapshot| {
                    snapshot
                        .providers()
                        .iter()
                        .map(|provider| provider.provider_id.clone())
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}
