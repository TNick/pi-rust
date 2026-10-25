//! The single atomic decision that closes a generation to new calls.
//!
//! Supersession and call entry are decided on one packed word so that a host can
//! never observe a generation it already replaced accept a new call, and so a
//! refused entry leaves no trace in the in-flight total a host watches while it
//! drains that generation. Two separate flags would admit a window: a caller
//! that read "still active" a moment before the retire landed would report a
//! call that is really addressed to a superseded session.

use std::sync::atomic::{AtomicU64, Ordering};

/// Tag bit: the generation no longer admits calls.
const CLOSED: u64 = 1 << 63;

/// Packed entry state of one generation: [`CLOSED`] plus the admitted-call count.
///
/// The count occupies the low 63 bits, so it can only reach the tag bit after
/// 2^63 concurrently admitted calls — unreachable for a per-call lease — and
/// `+ 1` / `- 1` never disturb the tag.
///
/// Private members:
/// - [`Admissions::state`]: the packed closed-flag plus count word.
pub(crate) struct Admissions {
    state: AtomicU64,
}

impl Admissions {
    /// Build the state of a freshly published generation.
    ///
    /// # Returns
    /// Open, with no admitted call.
    pub(crate) fn open() -> Self {
        Self {
            state: AtomicU64::new(0),
        }
    }

    /// Whether this state still admits calls.
    ///
    /// # Returns
    /// `false` once [`Admissions::close`] ran.
    pub(crate) fn is_open(&self) -> bool {
        self.state.load(Ordering::SeqCst) & CLOSED == 0
    }

    /// Admit one call, unless the generation was closed.
    ///
    /// The CAS is the linearization point shared with [`Admissions::close`]: an
    /// entry that wins it is counted, and an entry that arrives after the close
    /// is refused without touching the count, so a caller never runs against a
    /// closed generation and a refused caller never appears in
    /// [`Admissions::admitted`].
    ///
    /// # Returns
    /// `true` when the call was admitted and counted.
    pub(crate) fn admit(&self) -> bool {
        let mut state = self.state.load(Ordering::SeqCst);
        loop {
            if state & CLOSED != 0 {
                return false;
            }
            match self.state.compare_exchange_weak(
                state,
                state + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(seen) => state = seen,
            }
        }
    }

    /// Release one admitted call.
    ///
    /// Called by the guard that [`Admissions::admit`] handed out; the admitted
    /// count is never decremented below the value the close observed.
    pub(crate) fn release(&self) {
        self.state.fetch_sub(1, Ordering::SeqCst);
    }

    /// Calls admitted and not yet released.
    ///
    /// # Returns
    /// The admitted-call count, excluding the closed tag.
    pub(crate) fn admitted(&self) -> usize {
        (self.state.load(Ordering::SeqCst) & !CLOSED) as usize
    }

    /// Stop admitting new calls, keeping the admitted count intact.
    ///
    /// # Returns
    /// `true` only for the open-to-closed transition, so the caller runs the
    /// supersession side effects exactly once.
    pub(crate) fn close(&self) -> bool {
        let mut state = self.state.load(Ordering::SeqCst);
        loop {
            if state & CLOSED != 0 {
                return false;
            }
            match self.state.compare_exchange_weak(
                state,
                state | CLOSED,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(seen) => state = seen,
            }
        }
    }
}
