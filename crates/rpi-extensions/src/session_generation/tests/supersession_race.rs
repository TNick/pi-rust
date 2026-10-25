//! Supersession and call entry must be one atomic decision.
//!
//! A publication holds the gate while the host commits its catalog rows; entry
//! attempts run against the generation the slot publishes. The defect these tests
//! pin had two faces: the gate released the swapped-out generation before
//! retiring it, so a call could enter a session the host had already replaced;
//! and entry published its count before checking staleness, so a generation
//! being drained reported a call that was really refused.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

use crate::session_generation::{ExtensionGenerationSlot, GenerationCandidate, GenerationId};

use super::helpers::session_with_tool;

/// Publications the publisher performs while readers try to enter.
const PUBLICATIONS: u64 = 250;

/// Reader threads hammering entry while publications land.
const READERS: usize = 6;

/// How long the commit hook keeps the gate busy, so readers are parked on the
/// gate and wake exactly when the publication releases it.
const GATE_HOLD: Duration = Duration::from_micros(20);

#[test]
fn a_publication_closes_entry_before_the_replacement_is_visible() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("g1", "/plugins/g1.so"));
    let stop = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(READERS + 1));
    let violations = Arc::new(Mutex::new(Vec::<String>::new()));
    let refused = Arc::new(AtomicUsize::new(0));

    let readers: Vec<_> = (0..READERS)
        .map(|_| {
            let slot = slot.clone();
            let stop = Arc::clone(&stop);
            let start = Arc::clone(&start);
            let violations = Arc::clone(&violations);
            let refused = Arc::clone(&refused);
            std::thread::spawn(move || {
                start.wait();
                while !stop.load(Ordering::SeqCst) {
                    // A host, tool adapter, or drain handle addresses the lease it
                    // captured directly instead of going through the slot, which is
                    // how plugin tool dispatch reaches a generation.
                    let lease = slot.current();
                    let observed = slot.current_id();
                    match lease.begin_call() {
                        Ok(guard) => {
                            if observed != lease.id() {
                                violations.lock().unwrap().push(format!(
                                    "read {} observed live generation {} and still \
                                     entered generation {}",
                                    std::thread::current().name().unwrap_or("reader"),
                                    observed,
                                    lease.id()
                                ));
                            }
                            drop(guard);
                        }
                        Err(_) => {
                            refused.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            })
        })
        .collect();

    start.wait();
    let retired: Vec<Arc<crate::SessionGeneration>> = (2..=PUBLICATIONS)
        .map(|index| {
            let name = format!("g{index}");
            let published = slot
                .publish_with(
                    GenerationCandidate::new(session_with_tool(
                        &name,
                        &format!("/plugins/{name}.so"),
                    )),
                    |_| {
                        std::thread::sleep(GATE_HOLD);
                        Ok(None)
                    },
                )
                .expect("the commit accepts the candidate");
            published.previous
        })
        .collect();
    stop.store(true, Ordering::SeqCst);
    for reader in readers {
        reader.join().expect("reader completes");
    }

    let violations = violations.lock().unwrap().clone();
    assert!(
        violations.is_empty(),
        "a superseded generation admitted {} new call(s): {}",
        violations.len(),
        violations.join("; ")
    );
    assert!(
        refused.load(Ordering::SeqCst) > 0,
        "the readers must actually collide with supersession for this to prove anything"
    );
    assert_eq!(slot.current_id(), GenerationId::new(PUBLICATIONS));
    for previous in &retired {
        assert!(
            !previous.is_active(),
            "generation {} is still marked active after being superseded",
            previous.id()
        );
        assert_eq!(
            previous.in_flight(),
            0,
            "generation {} still reports an admitted call after every guard was released",
            previous.id()
        );
    }
}

#[test]
fn a_refused_entry_never_appears_in_the_drained_total() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("g1", "/plugins/g1.so"));
    let published = slot
        .publish(GenerationCandidate::new(session_with_tool(
            "g2",
            "/plugins/g2.so",
        )))
        .expect("publication is unconditional");
    let retired = Arc::clone(&published.previous);

    let stop = Arc::new(AtomicBool::new(false));
    let phantom = Arc::new(AtomicUsize::new(0));
    let admitted = Arc::new(AtomicUsize::new(0));

    let observer = {
        let retired = Arc::clone(&retired);
        let stop = Arc::clone(&stop);
        let phantom = Arc::clone(&phantom);
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                if retired.in_flight() > 0 {
                    phantom.fetch_add(1, Ordering::SeqCst);
                }
            }
        })
    };
    let caller = {
        let retired = Arc::clone(&retired);
        let stop = Arc::clone(&stop);
        let admitted = Arc::clone(&admitted);
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                if let Ok(guard) = retired.begin_call() {
                    admitted.fetch_add(1, Ordering::SeqCst);
                    drop(guard);
                }
            }
        })
    };

    std::thread::sleep(Duration::from_millis(20));
    stop.store(true, Ordering::SeqCst);
    observer.join().expect("observer completes");
    caller.join().expect("caller completes");

    assert_eq!(
        admitted.load(Ordering::SeqCst),
        0,
        "a retired generation admitted a new call"
    );
    assert_eq!(
        phantom.load(Ordering::SeqCst),
        0,
        "a refused entry was published in the drained total, so a host waiting for \
         the generation to drain would wait on a call that does not exist"
    );
    assert!(retired.is_idle());
}
