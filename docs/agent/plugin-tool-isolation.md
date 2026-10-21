# Plugin Tool Isolation Boundary

## Current execution model

`rpi-extensions` loads native extensions as in-process dynamic libraries. Their
tool callbacks receive host-provided arguments and run through a blocking
adapter. The blocking adapter handles async scheduling, cancellation, and
library lifetime; it does not isolate memory, filesystem access, credentials,
or process authority.

The host authorization callback runs before a registered tool callback. A
denial prevents that callback from running, but authorization does not stop
other code in the same extension from accessing resources available to the
host process. Hosts that require mediation should build registered tools with
`PluginToolAdapter::from_registered` and install a host-owned authorization
callback. Registered adapters fail closed until that callback is present.
`PluginToolAdapter::new` remains a compatibility constructor for direct,
unattributed adapters and does not provide this gate.

## Requirements for an isolated worker

An isolated implementation needs a separate worker process and a host-owned
IPC broker. It must not load the extension library into the canonical host
process. The worker may receive a tool schema and a bounded invocation frame,
but it must not receive provider credentials, direct storage handles, or
host-side capability tokens that authorize arbitrary operations.

For each invocation, the host must validate the registered extension origin,
tool schema, argument size, canonical branch identity, and ScribeRig
capabilities before sending an invocation to the worker. Any semantic
operation must return over a typed IPC request and pass through the host's
capability and canonical-state checks before it is applied. The worker's
response is untrusted data and must be bounded and validated before it enters
the agent result stream.

The process boundary also needs OS-enforced resource restrictions that deny
the worker direct access to canonical storage and host credentials. A child
process running with the host user's ordinary permissions is not sufficient.
The transport needs request identifiers, cancellation, deadlines, frame-size
limits, worker-crash handling, and a versioned protocol. Until these controls
exist, native plugin tools remain in-process extensions and must not be
described as sandboxed or isolated.
