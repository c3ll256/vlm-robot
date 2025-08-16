import websocket
import json
import time
import threading
import gradio as gr
import queue
from datetime import datetime

# WebSocket URL 和 headers（请替换 token）
url = "wss://autoglm-api.zhipuai.cn/openapi/v1/autoglm/developer"
headers = {
    "Authorization": "Bearer hWve-TpbFwXTbzRbrSo0VPvB5_1ngXJo6Kv9tflU5cgGnZz27QxKvY-kLVpvQV2inrvd1dH5ahviXfmxS2O2eA.IOM7RTz02IE0hP8rUJ-B__twAeohf6N7xN6c078yGRI"  # 替换为真实 token
}

# 全局变量
ws = None
message_queue = queue.Queue()
chat_history = []

def format_timestamp():
    """格式化当前时间戳"""
    return datetime.now().strftime("%H:%M:%S")

def add_to_chat_history(sender, message, msg_type="normal"):
    """添加消息到聊天历史"""
    timestamp = format_timestamp()
    if msg_type == "system":
        formatted_msg = f"[{timestamp}] 🔧 {message}"
    elif sender == "用户":
        formatted_msg = f"[{timestamp}] 👤 **{sender}**: {message}"
    else:
        formatted_msg = f"[{timestamp}] 🤖 **{sender}**: {message}"
    
    chat_history.append(formatted_msg)
    return "\n\n".join(chat_history)

# WebSocket 事件回调函数
def on_message(ws, message):
    print("Received:", message)
    try:
        # 解析JSON消息
        data = json.loads(message)
        msg_type = data.get("msg_type", "unknown")
        
        if msg_type == "server_init":
            message_queue.put(("system", "服务器初始化完成"))
        elif msg_type == "server_session":
            biz_type = data.get("data", {}).get("biz_type", "")
            if biz_type == "init_vm":
                vm_state = data.get("data", {}).get("vm_state", "")
                message_queue.put(("system", f"虚拟机状态: {vm_state}"))
            elif biz_type == "init_session":
                vm_id = data.get("data", {}).get("vm_id", "")
                message_queue.put(("system", f"会话初始化完成，VM ID: {vm_id[:20]}..."))
        elif msg_type == "client_test":
            instruction = data.get("data", {}).get("instruction", "")
            message_queue.put(("系统", f"任务已接收: {instruction}"))
        elif msg_type == "server_task":
            biz_type = data.get("data", {}).get("biz_type", "")
            data_agent = data.get("data", {}).get("data_agent", {})
            action = data_agent.get("action", "")
            
            if biz_type == "task":
                if action == "launch":
                    app_name = data_agent.get("app_name", "")
                    message_queue.put(("AI助手", f"正在启动应用: {app_name}"))
                elif action == "tap":
                    message_queue.put(("AI助手", "正在执行点击操作"))
                elif action == "swipe":
                    message_queue.put(("AI助手", "正在执行滑动操作"))
            elif biz_type == "interact":
                interact_data = data_agent.get("interact", [])
                if interact_data:
                    interact_info = interact_data[0].get("title", "交互")
                    message_queue.put(("AI助手", f"需要用户交互: {interact_info}"))
            elif biz_type == "take_over":
                take_over_msg = data_agent.get("message", "")
                message_queue.put(("AI助手", f"需要接管: {take_over_msg}"))
        else:
            # 显示原始消息
            message_queue.put(("系统", f"收到消息类型: {msg_type}"))
            
    except json.JSONDecodeError:
        message_queue.put(("系统", f"收到非JSON消息: {message[:100]}..."))
    except Exception as e:
        message_queue.put(("系统", f"处理消息时出错: {str(e)}"))

def on_open(ws):
    print("WebSocket connection opened.")
    message_queue.put(("system", "WebSocket 连接已建立"))

def on_error(ws, error):
    print("Error:", error)
    message_queue.put(("system", f"连接错误: {str(error)}"))

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")
    message_queue.put(("system", "WebSocket 连接已关闭"))

def build_message(instruction):
    """构建要发送的 JSON 消息"""
    return {
        "timestamp": int(time.time() * 1000),
        "conversation_id": "",
        "msg_type": "client_test",
        "msg_id": "",
        "data": {
            "biz_type": "test_agent",
            "instruction": instruction
        }
    }

def start_websocket():
    """启动 WebSocket 连接"""
    global ws
    try:
        ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()
    except Exception as e:
        print(f"WebSocket error: {e}")
        message_queue.put(("system", f"WebSocket 连接错误: {str(e)}"))

def send_message(message, history):
    """发送消息并更新聊天历史"""
    if not message.strip():
        return update_chat(), ""
    
    # 添加用户消息到历史
    add_to_chat_history("用户", message)
    
    try:
        if ws and hasattr(ws, 'sock') and ws.sock and ws.sock.connected:
            msg = build_message(message)
            ws.send(json.dumps(msg))
            print("Sent:", msg)
            add_to_chat_history("系统", "消息已发送", "system")
        else:
            add_to_chat_history("系统", "WebSocket 未连接，正在尝试重连...", "system")
            # 尝试重新连接
            threading.Thread(target=start_websocket, daemon=True).start()
            
    except Exception as e:
        add_to_chat_history("系统", f"发送消息时出错: {str(e)}", "system")
        print(f"Send message error: {e}")
    
    # 处理队列中的新消息并返回更新的聊天历史
    while not message_queue.empty():
        try:
            sender, msg = message_queue.get_nowait()
            if sender == "system":
                add_to_chat_history("系统", msg, "system")
            else:
                add_to_chat_history(sender, msg)
        except queue.Empty:
            break
    
    return "\n\n".join(chat_history), ""

def update_chat():
    """更新聊天显示（处理来自WebSocket的消息）"""
    updated = False
    while not message_queue.empty():
        try:
            sender, message = message_queue.get_nowait()
            if sender == "system":
                add_to_chat_history("系统", message, "system")
            else:
                add_to_chat_history(sender, message)
            updated = True
        except queue.Empty:
            break
    
    if updated:
        return "\n\n".join(chat_history)
    return gr.update()

def connect_websocket():
    """连接WebSocket"""
    global ws
    if ws and hasattr(ws, 'sock') and ws.sock and ws.sock.connected:
        return add_to_chat_history("系统", "WebSocket 已经连接", "system"), "\n\n".join(chat_history)
    
    add_to_chat_history("系统", "正在连接 WebSocket...", "system")
    threading.Thread(target=start_websocket, daemon=True).start()
    return "\n\n".join(chat_history)

def disconnect_websocket():
    """断开WebSocket连接"""
    global ws
    try:
        if ws:
            ws.close()
        add_to_chat_history("系统", "WebSocket 连接已断开", "system")
    except Exception as e:
        add_to_chat_history("系统", f"断开连接时出错: {str(e)}", "system")
    return "\n\n".join(chat_history)

def clear_chat():
    """清空聊天历史"""
    global chat_history
    chat_history = []
    return ""

# 创建 Gradio 界面
with gr.Blocks(
    title="AutoGLM 对话客户端",
    theme=gr.themes.Soft(),
    css="""
    .message-container {
        max-height: 600px;
        overflow-y: auto;
    }
    """
) as app:
    gr.Markdown(
        """
        # 🤖 AutoGLM 对话客户端
        
        这是一个基于 WebSocket 的 AutoGLM 客户端，可以发送指令并实时查看响应。
        
        ### 使用说明：
        1. 点击 "连接 WebSocket" 建立连接
        2. 在输入框中输入您的指令
        3. 点击 "发送" 或按回车键发送消息
        4. 在聊天区域查看实时响应
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            connect_btn = gr.Button("🔗 连接 WebSocket", variant="primary")
            disconnect_btn = gr.Button("🔌 断开连接", variant="secondary")
            refresh_btn = gr.Button("🔄 刷新消息", variant="secondary")
            clear_btn = gr.Button("🗑️ 清空聊天", variant="secondary")
        
        with gr.Column(scale=3):
            # 聊天历史显示区域
            chat_display = gr.Textbox(
                label="💬 聊天历史",
                value="",
                lines=20,
                max_lines=30,
                interactive=False,
                elem_classes=["message-container"]
            )
    
    with gr.Row():
        with gr.Column(scale=4):
            message_input = gr.Textbox(
                label="✏️ 输入消息",
                placeholder="请输入您的指令，例如：帮我点一杯奶茶",
                lines=2,
                max_lines=5
            )
        with gr.Column(scale=1, min_width=120):
            send_btn = gr.Button("📤 发送", variant="primary", size="lg")
    
    # 示例消息
    gr.Examples(
        examples=[
            ["帮我点一杯奶茶"],
            ["帮我在小红书找三篇云南的旅游攻略"],
            ["帮我查看今天的天气"],
            ["帮我打开淘宝搜索手机"],
        ],
        inputs=message_input,
        label="💡 示例指令"
    )
    
    # 事件绑定
    connect_btn.click(
        connect_websocket,
        outputs=[chat_display]
    )
    
    disconnect_btn.click(
        disconnect_websocket,
        outputs=[chat_display]
    )
    
    refresh_btn.click(
        update_chat,
        outputs=[chat_display]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chat_display]
    )
    
    send_btn.click(
        send_message,
        inputs=[message_input, chat_display],
        outputs=[chat_display, message_input]
    )
    
    message_input.submit(
        send_message,
        inputs=[message_input, chat_display],
        outputs=[chat_display, message_input]
    )
    
    # 自动连接 WebSocket
    app.load(
        connect_websocket,
        outputs=[chat_display]
    )

if __name__ == "__main__":
    # 启动应用
    print("正在启动 AutoGLM 对话客户端...")
    print("请在浏览器中打开显示的地址")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )