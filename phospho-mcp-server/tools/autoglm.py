import json
import threading
import time
from datetime import datetime
from typing import List, Optional

import websocket  # Requires websocket-client


def _format_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _build_message(instruction: str) -> dict:
    return {
        "timestamp": int(time.time() * 1000),
        "conversation_id": "",
        "msg_type": "client_test",
        "msg_id": "",
        "data": {
            "biz_type": "test_agent",
            "instruction": instruction,
        },
    }


def run_autoglm(
    instruction: str,
    token: str,
    url: str = "wss://autoglm-api.zhipuai.cn/openapi/v1/autoglm/developer",
    timeout_sec: int = 60,
) -> str:
    """
    Run a single AutoGLM session by sending an instruction and collecting outputs
    for up to timeout_sec seconds. Returns a human-readable transcript string.
    """

    transcript: List[str] = []
    stop_event = threading.Event()
    ws_ref: dict[str, Optional[websocket.WebSocketApp]] = {"ws": None}

    def add(sender: str, message: str) -> None:
        ts = _format_timestamp()
        transcript.append(f"[{ts}] {sender}: {message}")

    def on_open(ws):  # type: ignore
        add("system", "WebSocket 已连接，正在提交指令…")
        try:
            ws.send(json.dumps(_build_message(instruction)))
            add("system", "指令已发送")
        except Exception as e:  # pragma: no cover
            add("system", f"发送失败: {e}")

    def on_message(ws, message):  # type: ignore
        try:
            data = json.loads(message)
        except Exception:
            add("server", f"原始消息: {str(message)[:200]}")
            return

        msg_type = data.get("msg_type", "unknown")
        if msg_type == "server_init":
            add("system", "服务器初始化完成")
            return
        if msg_type == "server_session":
            biz_type = data.get("data", {}).get("biz_type", "")
            if biz_type == "init_vm":
                vm_state = data.get("data", {}).get("vm_state", "")
                add("system", f"虚拟机状态: {vm_state}")
            elif biz_type == "init_session":
                vm_id = data.get("data", {}).get("vm_id", "")
                add("system", f"会话初始化完成，VM ID: {vm_id[:20]}…")
            return
        if msg_type == "client_test":
            instruction_text = data.get("data", {}).get("instruction", "")
            add("server", f"任务已接收: {instruction_text}")
            return
        if msg_type == "server_task":
            biz_type = data.get("data", {}).get("biz_type", "")
            data_agent = data.get("data", {}).get("data_agent", {})
            action = data_agent.get("action", "")
            if biz_type == "task":
                if action == "launch":
                    app_name = data_agent.get("app_name", "")
                    add("autoglm", f"正在启动应用: {app_name}")
                elif action == "tap":
                    add("autoglm", "正在执行点击操作")
                elif action == "swipe":
                    add("autoglm", "正在执行滑动操作")
            elif biz_type == "interact":
                interact_data = data_agent.get("interact", [])
                if interact_data:
                    interact_info = interact_data[0].get("title", "交互")
                    add("autoglm", f"需要用户交互: {interact_info}")
            elif biz_type == "take_over":
                take_over_msg = data_agent.get("message", "")
                add("autoglm", f"需要接管: {take_over_msg}")
            return

        # Fallback: show type
        add("server", f"收到消息类型: {msg_type}")

    def on_error(ws, error):  # type: ignore
        add("system", f"连接错误: {error}")

    def on_close(ws, code, reason):  # type: ignore
        add("system", "WebSocket 已关闭")
        stop_event.set()

    headers = {"Authorization": f"Bearer {token}"}

    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws_ref["ws"] = ws

    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()

    # Wait with timeout, then close
    start = time.time()
    while time.time() - start < max(1, timeout_sec):
        if stop_event.wait(timeout=0.1):
            break
        # continue collecting until timeout

    try:
        if ws_ref["ws"] is not None:
            ws_ref["ws"].close()
    except Exception:
        pass

    return "\n".join(transcript)


