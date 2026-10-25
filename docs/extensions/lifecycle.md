# 扩展生命周期事件

本文描述当前实现，替代旧的生命周期事件总线设计稿。事件标签和 ABI 定义见 [rpi-plugin-sdk](../../crates/rpi-plugin-sdk/src/lib.rs)，分发实现见 [translate.rs](../../crates/rpi-extensions/src/translate.rs)。

`BeforeTuiStart` 是 rpi 特有的启动事件，在 TUI 初始化前调用。扩展可通过返回值否决启动。`SessionStart` 和 `SessionShutdown` 处理会话启动与关闭。生命周期事件使用无 payload 的 `EventEmpty`，不是旧设计稿中的 Rust 资源注入总线。

宿主在 blocking 线程中调用 handler，并在 FFI 边界捕获 panic。分发使用以下超时预算：

| 事件 | 单个 handler 的超时 |
| --- | --- |
| `BeforeTuiStart` | 5 秒 |
| `SessionShutdown` | 15 秒 |
| 其他生命周期标签 | 10 秒 |

超时后记录日志并跳过等待；已经运行的 blocking handler 仍可能继续执行，超时不会强制终止它。扩展应自行限制阻塞和资源占用，且不能让 panic 穿过 ABI 边界。

扩展注册、支持的事件与工具生命周期见 [扩展开发指南](authoring.md)。

## 会话世代替换（宿主侧）

宿主重新加载扩展时需要同时替换三样东西：已加载的 `ExtensionSession`、由它派生的模型目录行，以及让旧会话失效的陈旧状态。逐项替换会留下窗口：调用方可能把新目录和旧会话配对，或调用一个正在被卸载的插件。`rpi-extensions` 用 `ExtensionGenerationSlot` 作为唯一的发布闸口：

- `GenerationCandidate` 暂存新加载的会话，`retain` 可把宿主自有的替换资源（例如隔离 worker 的客户端句柄）绑定到同一生代的寿命上。
- `publish_with` 在持有发布闸口期间执行宿主提交的回调，并记录它返回的 catalog revision，使每个世代 id 对应一次目录提交；提交被拒绝时当前世代保持不变（仍可接受新的 `begin_call`），候选会话的产物按未发布释放。
- 读取方要么看到完整的旧世代，要么看到完整的新世代，不会看到半替换状态。
- 取代与进入调用是一次原子决策：被替换的世代在同一个闸口临界区内、新世代可见之前就被 retire，因此不存在“新目录已生效、旧会话仍接受新调用”的窗口；`begin_call` 与 retire 是世代 admission 状态上的单次 CAS，赢出的进入会被计数，retire 之后到达的进入被拒绝且不会在排空计数中留下痕迹。
- 被替换的世代进入 retired：不再接受新的 `begin_call`，其 `RegistrySnapshot` 与 `ActionBridge` 的共享陈旧标记被置位；它的 keepalive 与 pinned 资源在所有租约（含运行中的 `InFlightGuard`）释放前保持有效，因此已进入插件代码的调用不会失去函数指针。`PublishedGeneration::previous` 就是用来观察 `in_flight` 排空的句柄。
- `GenerationId` 与 `GenerationIdentity` 只包含 id、计数、工具名、provider id 与 catalog revision，不含插件路径、`apiKeyEnv`、`base_url` 或任何凭据，可安全记录或回传。

宿主必须在发布之前完成仍需旧会话存活的工作（例如 `SessionShutdown` 事件），因为发布就是旧世代停止接受新调度的时刻；retire 的副作用（置位 `RegistrySnapshot` 标记、`ActionBridge::invalidate` 及其触发的宿主取消回调）也在闸口内执行，因此发布路径上的宿主代码不能回调进这个 slot。插件主动请求重载的信号仍是 `ReloadMailbox`。
