//! Tests for atomic extension-session generation replacement.

mod admission_state;
mod helpers;
mod secret_free_output;
mod supersession_race;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::ActionBridge;

use super::{
    ExtensionGenerationSlot, GenerationCandidate, GenerationError, GenerationId, GenerationIdentity,
};

use helpers::{session_with_bridge, session_with_provider, session_with_tool, Pinned, UnusedHost};

#[test]
fn publication_replaces_the_live_generation() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("first", "/plugins/first.so"));
    assert_eq!(slot.current_id(), GenerationId::new(1));
    assert_eq!(slot.current_identity().tools, vec!["first".to_owned()]);

    let published = slot
        .publish(GenerationCandidate::new(session_with_tool(
            "second",
            "/plugins/second.so",
        )))
        .expect("publication is unconditional");

    assert_eq!(slot.current_id(), GenerationId::new(2));
    assert_eq!(published.current.id(), slot.current_id());
    assert_eq!(published.previous.id(), GenerationId::new(1));
    assert_eq!(slot.current_identity().tools, vec!["second".to_owned()]);
    assert!(published.current.is_active());
    assert!(!published.previous.is_active());
}

#[test]
fn a_superseded_generation_refuses_new_calls() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("first", "/plugins/first.so"));
    let first_id = slot.current_id();
    let published = slot
        .publish(GenerationCandidate::new(session_with_tool(
            "second",
            "/plugins/second.so",
        )))
        .expect("publication is unconditional");

    assert_eq!(
        published.previous.begin_call().err(),
        Some(GenerationError::Retired {
            generation: first_id
        }),
        "a retired generation must not accept a new call"
    );
    assert_eq!(published.previous.in_flight(), 0);
    assert!(
        !published
            .previous
            .snapshot()
            .expect("the test session registers a tool")
            .is_active(),
        "the superseded snapshot must carry the shared staleness state"
    );
    let guard = slot
        .begin_call()
        .expect("the live generation admits a call");
    assert_eq!(guard.id(), slot.current_id());
    assert_eq!(slot.current().in_flight(), 1);
    drop(guard);
    assert!(slot.current().is_idle());
}

#[test]
fn an_in_flight_call_keeps_the_superseded_generation_resources_alive() {
    let dropped = Arc::new(AtomicBool::new(false));
    let slot = ExtensionGenerationSlot::empty();
    let first = slot
        .publish(
            GenerationCandidate::new(session_with_tool("first", "/plugins/first.so")).retain(
                Arc::new(Pinned {
                    dropped: Arc::clone(&dropped),
                }),
            ),
        )
        .expect("the first session is published through a candidate");
    let guard = first
        .current
        .begin_call()
        .expect("live generation admits a call");
    let keepalive = Arc::downgrade(&first.current.keepalive());

    let published = slot
        .publish(GenerationCandidate::new(session_with_tool(
            "second",
            "/plugins/second.so",
        )))
        .expect("publication is unconditional");

    assert_eq!(published.previous.in_flight(), 1);
    assert_eq!(
        guard.identity().tools,
        vec!["first".to_owned()],
        "the running call still reads the session it entered"
    );
    assert!(
        keepalive.upgrade().is_some(),
        "plugin artifacts stay mapped while a call is running"
    );
    assert!(
        !dropped.load(Ordering::SeqCst),
        "the pinned replacement resource survives the swap"
    );

    drop(guard);
    assert!(published.previous.is_idle());
    drop(published.previous);
    drop(first.current);
    assert!(
        keepalive.upgrade().is_none(),
        "artifacts are released once no lease references the generation"
    );
    assert!(
        dropped.load(Ordering::SeqCst),
        "the pinned resource is released with the generation"
    );
}

#[test]
fn a_republished_session_keeps_its_resources_until_the_new_generation_retires() {
    let dropped = Arc::new(AtomicBool::new(false));
    let slot = ExtensionGenerationSlot::empty();
    let initial = slot
        .publish(
            GenerationCandidate::new(session_with_tool("retained", "/plugins/retained.so")).retain(
                Arc::new(Pinned {
                    dropped: Arc::clone(&dropped),
                }),
            ),
        )
        .expect("publish initial generation");
    drop(initial.previous);
    let first = slot.current();
    drop(initial.current);

    let refreshed = slot
        .publish(GenerationCandidate::from_generation(&first))
        .expect("republish current session");
    assert_eq!(refreshed.current.identity().tools, vec!["retained"]);
    assert_ne!(refreshed.current.id(), first.id());
    assert!(
        !dropped.load(Ordering::SeqCst),
        "the new generation must retain the current generation's resources"
    );

    drop(first);
    drop(refreshed.previous);
    let retired = slot
        .publish(GenerationCandidate::new(crate::ExtensionSession::none()))
        .expect("replace retained session");
    drop(retired.previous);
    drop(retired.current);
    drop(refreshed.current);
    drop(slot);
    assert!(
        dropped.load(Ordering::SeqCst),
        "retained resources are released after the replacement generation retires"
    );
}

#[test]
fn resolving_a_superseded_generation_id_fails_closed() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("first", "/plugins/first.so"));
    let first_id = slot.current_id();
    let published = slot
        .publish(GenerationCandidate::new(session_with_tool(
            "second",
            "/plugins/second.so",
        )))
        .expect("publication is unconditional");

    assert_eq!(
        slot.resolve(first_id).map(|lease| lease.id()),
        Err(GenerationError::Stale {
            generation: first_id,
            current: published.current.id(),
        })
    );
    assert_eq!(
        slot.resolve(published.current.id()).map(|lease| lease.id()),
        Ok(published.current.id())
    );
}

#[test]
fn a_rejected_commit_changes_nothing() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("first", "/plugins/first.so"));
    let live_id = slot.current_id();
    let guard = slot
        .begin_call()
        .expect("the live generation admits a call");

    let error = slot
        .publish_with(
            GenerationCandidate::new(session_with_tool("second", "/plugins/second.so")),
            |_| Err("package root is not authorized".to_owned()),
        )
        .expect_err("a rejected commit must fail the publication");
    assert_eq!(
        error,
        GenerationError::CommitRejected {
            reason: "package root is not authorized".to_owned()
        }
    );

    assert_eq!(slot.current_id(), live_id);
    assert_eq!(slot.resolve(live_id).map(|lease| lease.id()), Ok(live_id));
    assert_eq!(slot.current_identity().tools, vec!["first".to_owned()]);
    assert!(
        slot.begin_call().is_ok(),
        "a rejected commit must not retire the generation that is still live"
    );
    assert_eq!(guard.in_flight(), 1, "the live call is unaffected");
    drop(guard);
}

#[test]
fn the_commit_hook_pairs_the_new_identity_with_a_catalog_revision() {
    let observed: Arc<Mutex<Vec<GenerationIdentity>>> = Arc::new(Mutex::new(Vec::new()));
    let slot = ExtensionGenerationSlot::empty();

    let published = slot
        .publish_with(
            GenerationCandidate::new(session_with_provider()),
            |identity| {
                observed.lock().unwrap().push(identity.clone());
                Ok(Some(41))
            },
        )
        .expect("the host commit accepts the candidate");

    let seen = observed.lock().unwrap()[0].clone();
    assert_eq!(seen.id, published.current.id());
    assert_eq!(seen.id, GenerationId::new(2));
    assert_eq!(
        seen.catalog_revision, None,
        "the hook runs before the revision exists"
    );
    assert_eq!(seen.providers, vec!["pluggy".to_owned()]);
    assert_eq!(published.current.catalog_revision(), Some(41));
    assert_eq!(slot.current_identity().catalog_revision, Some(41));
    assert_eq!(
        slot.resolve(seen.id)
            .expect("the committed identity is the live generation")
            .catalog_revision(),
        Some(41)
    );
}

#[test]
fn the_commit_runs_before_the_replacement_is_visible() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("first", "/plugins/first.so"));
    let previous = slot.current();

    let published = slot
        .publish_with(
            GenerationCandidate::new(session_with_tool("second", "/plugins/second.so")),
            |identity| {
                assert_eq!(
                    identity.tools,
                    vec!["second".to_owned()],
                    "the hook is told the identity being published"
                );
                assert!(
                    previous.is_active(),
                    "the superseded session is still live while the commit runs"
                );
                assert_eq!(previous.identity().tools, vec!["first".to_owned()]);
                Ok(Some(7))
            },
        )
        .expect("the host commit accepts the candidate");

    assert_eq!(published.current.catalog_revision(), Some(7));
    assert!(!previous.is_active());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retirement_invalidates_the_superseded_action_bridge() {
    let bridge = ActionBridge::new(tokio::runtime::Handle::current(), Arc::new(UnusedHost));
    let slot = ExtensionGenerationSlot::new(session_with_bridge("first", &bridge));
    assert!(bridge.is_active());

    let published = slot
        .publish(GenerationCandidate::new(session_with_tool(
            "second",
            "/plugins/second.so",
        )))
        .expect("publication is unconditional");

    assert!(
        !bridge.is_active(),
        "plugin host actions on the superseded bridge must fail fast"
    );
    assert!(slot.begin_call().is_ok());
    drop(published);
}

#[test]
fn retiring_the_slot_stops_dispatch_until_a_replacement_lands() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("first", "/plugins/first.so"));
    let live_id = slot.current_id();

    let retired = slot.retire();
    assert!(!retired.is_active());
    assert_eq!(slot.retire().id(), live_id, "retirement is idempotent");
    assert!(slot.begin_call().is_err());
    assert_eq!(
        slot.resolve(live_id).map(|lease| lease.id()),
        Err(GenerationError::Retired {
            generation: live_id
        })
    );

    let republished = slot
        .publish(GenerationCandidate::new(session_with_tool(
            "third",
            "/plugins/third.so",
        )))
        .expect("a replacement can be published after retirement");
    assert!(republished.current.is_active());
    assert!(!republished.previous.is_active());
    assert!(slot.begin_call().is_ok());
}

#[test]
fn readers_never_observe_a_mixed_generation() {
    let slot = ExtensionGenerationSlot::new(session_with_tool("g1", "/plugins/g1.so"));
    let stop = Arc::new(AtomicBool::new(false));
    let start = Arc::new(std::sync::Barrier::new(2));
    let publisher = {
        let slot = slot.clone();
        let stop = Arc::clone(&stop);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            start.wait();
            for index in 2..=6u64 {
                let name = format!("g{index}");
                slot.publish(GenerationCandidate::new(session_with_tool(
                    &name,
                    &format!("/plugins/{name}.so"),
                )))
                .expect("publication is unconditional");
            }
            stop.store(true, Ordering::SeqCst);
        })
    };

    // Every read must see one whole generation: a session always belongs to the
    // id it was published under, never to a neighbour's.
    start.wait();
    while !stop.load(Ordering::SeqCst) {
        let lease = slot.current();
        assert_eq!(
            lease.identity().tools,
            vec![format!("g{}", lease.id().get())],
            "generation {} must expose only its own session",
            lease.id()
        );
    }
    publisher.join().expect("publisher completes");

    let final_lease = slot.current();
    assert_eq!(final_lease.id(), GenerationId::new(6));
    assert_eq!(final_lease.identity().tools, vec!["g6".to_owned()]);
}

#[test]
fn generation_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExtensionGenerationSlot>();
    assert_send_sync::<super::SessionGeneration>();
    assert_send_sync::<super::InFlightGuard>();
    assert_send_sync::<GenerationCandidate>();
    assert_send_sync::<super::PublishedGeneration>();
    assert_send_sync::<GenerationIdentity>();
    assert_send_sync::<GenerationError>();
}
