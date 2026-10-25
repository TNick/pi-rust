//! Atomic replacement of the live extension session as a numbered generation.
//!
//! A host that reloads extensions has to change three things together: the
//! loaded [`ExtensionSession`] (plugin artifacts + merged [`RegistrySnapshot`]),
//! the host rows derived from its registered providers, and the staleness state
//! that makes the superseded session refuse new work. Doing them one at a time
//! leaves a window where a caller can pair a new catalog with an old session, or
//! drive a plugin whose artifact is already being unloaded. This crate already
//! has the pieces of that window — [`PluginKeepalive`], the shared `active` flag
//! on [`RegistrySnapshot`], [`ActionBridge::invalidate`] — but no single owner
//! that publishes them together, so every host re-implemented the swap (the CLI's
//! reload routine is the only implementation at this pin).
//!
//! [`ExtensionGenerationSlot`] is that owner. It publishes one generation at a
//! time behind a single gate, so a reader either sees the whole old generation or
//! the whole new one:
//!
//! - **Staging.** [`GenerationCandidate`] carries a freshly loaded session plus
//!   any host-owned replacement resource the host wants pinned to the same
//!   lifetime (for an isolated-worker host, its worker client handle).
//! - **Atomic publication.** [`ExtensionGenerationSlot::publish_with`] runs the
//!   caller's commit hook — the place a host performs its model-catalog write —
//!   while the publication gate is held, and only then swaps the generation in.
//!   A rejected commit changes nothing: the live generation stays live and the
//!   candidate's artifacts are released unloaded.
//! - **Retention, not eager teardown.** Retirement flips staleness state only.
//!   The superseded generation's keepalive and pinned resources stay alive while
//!   any [`InFlightGuard`] or `Arc<SessionGeneration>` lease references them, so
//!   a call that already entered plugin code never loses its function pointers,
//!   and an isolated worker is not killed under a running tool call.
//!   [`PublishedGeneration::previous`] is returned precisely so a host can watch
//!   [`SessionGeneration::in_flight`] drain before it does something drastic.
//! - **Stale use is refused.** A retired generation admits no new
//!   [`InFlightGuard`], and [`ExtensionGenerationSlot::resolve`] turns a captured
//!   id from a superseded generation into [`GenerationError::Stale`] instead of
//!   handing back the live session.
//! - **Secret-free identity.** [`GenerationId`] and [`GenerationIdentity`] carry
//!   only ids, counts, tool names, provider ids, and the committed catalog
//!   revision — never a plugin path, credential field, or provider endpoint.
//!
//! ## Which upstream dispatch already observes staleness
//!
//! Retirement invalidates the superseded generation's [`RegistrySnapshot`] and
//! [`ActionBridge`], which is what the existing guards in `translate`,
//! `provider`, `resources`, `prompt_transform`, and `actions` already test.
//! Those guards refuse a *new* entry into a plugin callback; a callback already
//! running is not aborted, and plugin tool dispatch through
//! [`PluginToolAdapter`](crate::PluginToolAdapter) has no staleness gate at all —
//! it relies only on the keepalive. [`SessionGeneration::begin_call`] is the
//! missing host-side gate for that path: it refuses a call addressed to a
//! superseded generation and counts the ones in flight.
//!
//! ## Ordering a host must keep
//!
//! Publish *after* finishing anything that still needs the old session to look
//! live (shutdown lifecycle events, a final resource discover). Publication is
//! the point at which the old generation stops accepting new dispatch. The
//! plugin-initiated "reload me" signal stays
//! [`ReloadMailbox`](crate::ReloadMailbox); this module does not add a second
//! one.

mod generation;
mod identity;
mod slot;

#[cfg(test)]
mod tests;

pub use generation::{GenerationCandidate, GenerationError, InFlightGuard, SessionGeneration};
pub use identity::{GenerationId, GenerationIdentity};
pub use slot::{ExtensionGenerationSlot, PublishedGeneration};
