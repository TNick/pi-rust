use super::*;
use crate::registry::ExtensionRegistry;
use rpi_ai::types::{Api, StopReason};
use rpi_plugin_sdk::{ProviderStreamCompletionFn, ProviderStreamControl, ProviderStreamEmitFn};

#[derive(Default)]
struct PausedRequest {
    released: Mutex<bool>,
    wake: Condvar,
    rejected_after_cancel: AtomicBool,
}

extern "C" fn unused_request(_: StbStringRef, _: *mut StbString, _: *mut c_void) -> i32 {
    1
}

// Ignores the control surface and returns success only when the test releases
// it. The timeout bounds runtime shutdown even if an assertion fails.
extern "C" fn paused_success(
    _: StbStringRef,
    emit: Option<ProviderStreamEmitFn>,
    emit_data: *mut c_void,
    _: *const ProviderStreamControl,
    _: Option<ProviderStreamCompletionFn>,
    _: *mut c_void,
    out: *mut StbString,
    user_data: *mut c_void,
) -> i32 {
    // SAFETY: the test transfers one Arc into this single request invocation.
    let state = unsafe { Arc::from_raw(user_data.cast::<PausedRequest>()) };
    let emit = emit.expect("the host supplies an emit callback");
    for chunk in [
        serde_json::json!({"type":"usage", "usage": {
            "input":3,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":4,
            "cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0,"total":0.0}
        }}),
        serde_json::json!({"type":"text_delta","contentIndex":0,"delta":"hello"}),
    ] {
        emit(StbString::from_string(chunk.to_string()), emit_data);
    }
    let guard = state.released.lock().unwrap();
    let _wait = state
        .wake
        .wait_timeout_while(guard, Duration::from_secs(3), |released| !*released)
        .unwrap();
    let rc = emit(
        StbString::from_string(r#"{"type":"text_delta","contentIndex":0,"delta":" late"}"#.into()),
        emit_data,
    );
    state.rejected_after_cancel.store(rc != 0, Ordering::SeqCst);
    let message = serde_json::json!({
        "role":"assistant", "content":[{"type":"text","text":"hello late"}],
        "api":"faux", "provider":"probe", "model":"m", "usage":{
            "input":3,"output":2,"cacheRead":0,"cacheWrite":0,"totalTokens":5,
            "cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0,"total":0.0}
        }, "stopReason":"stop", "timestamp":0
    });
    // SAFETY: sole writer to the initialized ABI output slot.
    unsafe {
        *out = StbString::from_string(message.to_string());
    }
    0
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_finishes_before_an_uncooperative_plugin_and_retains_its_library() {
    let state = Arc::new(PausedRequest::default());
    let record = RegisteredProvider {
        provider_id: "probe".into(),
        base_url: "https://example".into(),
        api_style: "faux".into(),
        request_fn: unused_request,
        protected_request_fn: None,
        protected_stream_fn: None,
        stream_fn: Some(paused_success),
        plugin_free_string: rpi_plugin_sdk::callbacks::free_string,
        user_data: Arc::into_raw(Arc::clone(&state)) as *mut c_void,
    };
    let snapshot = Arc::new(ExtensionRegistry::new().snapshot());
    let keepalive = PluginKeepalive::empty();
    let weak = Arc::downgrade(&keepalive);
    let provider = PluggableProvider::new(record, snapshot, keepalive, Handle::current());
    let model = Model::new("m", "M", Api::Faux, "probe", "https://example");
    let options = SimpleStreamOptions::default();
    let scope = Arc::new(rpi_agent::tool_execution_scope::ToolExecutionScope::default());
    let mut stream = scope
        .run(provider.stream_simple(&model, &Context::new(vec![]), &options))
        .await;
    drop(provider);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match stream.next().await {
                Some(AssistantMessageEvent::TextDelta { .. }) => break,
                Some(_) => {}
                None => panic!("stream closed before the first delta"),
            }
        }
    })
    .await
    .expect("the plugin starts");
    options.signal.cancel();
    let terminal = tokio::time::timeout(Duration::from_millis(500), stream.next()).await;
    let library_retained = weak.upgrade().is_some();
    let capacity_retained = tokio::time::timeout(Duration::from_millis(25), scope.wait_idle())
        .await
        .is_err();

    // Release before asserting so a failure cannot strand a blocking pool thread.
    *state.released.lock().unwrap() = true;
    state.wake.notify_all();
    tokio::time::timeout(Duration::from_secs(2), async {
        while weak.upgrade().is_some() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the library is released after the FFI call finishes");

    assert!(library_retained, "the detached request must retain its DLL");
    assert!(
        capacity_retained,
        "an uncooperative provider must retain child capacity until its FFI call returns"
    );
    let terminal = terminal
        .expect("cancellation must not wait for plugin completion")
        .unwrap();
    let AssistantMessageEvent::Error { reason, error } = terminal else {
        panic!("a cancelled run must not report a successful Done");
    };
    assert_eq!(reason, ErrorReason::Aborted);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_eq!(error.content, vec![Content::text("hello")]);
    assert_eq!(error.model, "m");
    assert_eq!(error.provider, "probe");
    assert_eq!(error.usage.input, 3);
    assert_eq!(error.usage.output, 1);
    assert!(state.rejected_after_cancel.load(Ordering::SeqCst));
    assert!(
        stream.next().await.is_none(),
        "no output follows the aborted terminal"
    );
}

#[test]
fn cancellation_wakes_a_wait_already_in_progress() {
    let state = Arc::new(ProviderControlState::default());
    let other = Arc::clone(&state);
    let thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        other.cancel();
    });
    let start = std::time::Instant::now();
    assert!(state.wait(2_000));
    let elapsed = start.elapsed();
    thread.join().unwrap();
    assert!(
        elapsed < Duration::from_secs(1),
        "cancel wakeup took {elapsed:?}"
    );
}

fn mapper() -> StreamMapper {
    let mut mapper = StreamMapper::new(Api::Faux, "probe".into(), "m".into());
    mapper.apply(serde_json::json!({"type":"start"}));
    mapper.take_deferred();
    mapper
}

fn snapshot(content: serde_json::Value) -> serde_json::Value {
    let mut value =
        serde_json::to_value(AssistantMessage::empty(Api::Faux, "probe", "m", 0)).unwrap();
    value["content"] = serde_json::json!([content]);
    value
}

#[test]
fn delta_snapshots_override_text_thinking_and_tool_arguments() {
    let cases = [
        (
            "text_delta",
            serde_json::json!({"type":"text","text":"authoritative"}),
        ),
        (
            "thinking_delta",
            serde_json::json!({"type":"thinking","thinking":"authoritative"}),
        ),
        (
            "toolcall_delta",
            serde_json::json!({"type":"toolCall","id":"c1","name":"echo","arguments":{"value":"authoritative"}}),
        ),
    ];
    for (kind, content) in cases {
        let mut mapper = mapper();
        let partial = snapshot(content);
        let expected: AssistantMessage = serde_json::from_value(partial.clone()).unwrap();
        mapper
            .apply(serde_json::json!({"type":kind,"contentIndex":0,"delta":"x","partial":partial}));
        assert_eq!(mapper.accumulated().content, expected.content, "{kind}");
        if kind == "toolcall_delta" {
            let end = mapper
                .apply(serde_json::json!({"type":"toolcall_end","contentIndex":0}))
                .unwrap();
            let AssistantMessageEvent::ToolCallEnd { tool_call, .. } = end else {
                panic!()
            };
            assert_eq!(tool_call.arguments["value"], "authoritative");
        }
    }
}

#[test]
fn malformed_snapshots_fall_back_to_delta_accumulation() {
    let mut mapper = mapper();
    mapper.apply(
        serde_json::json!({"type":"text_delta","contentIndex":0,"delta":"hello","partial":{}}),
    );
    assert_eq!(mapper.accumulated().content, vec![Content::text("hello")]);
}
