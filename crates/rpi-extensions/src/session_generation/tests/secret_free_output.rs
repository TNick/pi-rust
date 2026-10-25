//! The generation-facing output must not disclose credentials, endpoints, or
//! plugin artifact paths.

use crate::ExtensionGenerationSlot;

use super::helpers::{session_with_provider, SECRET_ENV, SECRET_HOST, SECRET_PATH};

#[test]
fn generation_identity_and_debug_output_carry_no_secrets() {
    let slot = ExtensionGenerationSlot::new(session_with_provider());
    let identity = slot.current_identity();
    let lease = slot.current();
    let guard = slot
        .begin_call()
        .expect("the live generation admits a call");

    assert_eq!(identity.id, slot.current_id());
    assert_eq!(identity.loaded_plugins, 1);
    assert_eq!(identity.providers, vec!["pluggy".to_owned()]);

    let rendered = format!(
        "{identity:?} {} {:?} {:?} {}",
        identity.id,
        lease,
        guard,
        serde_json::to_string(&identity).expect("identity is serializable")
    );
    for secret in [SECRET_ENV, SECRET_HOST, SECRET_PATH] {
        assert!(
            !rendered.contains(secret),
            "generation output leaked {secret}: {rendered}"
        );
    }
}
