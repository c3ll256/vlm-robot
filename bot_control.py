import asyncio
import os
from pathlib import Path
from typing import Any, Dict
import argparse
import dotenv

dotenv.load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ai_agent import AIAgent
from task_executor import TaskExecutor

try:
    from zai import ZhipuAiClient  # 智谱 AI 官方 SDK
except Exception:  # pragma: no cover
    ZhipuAiClient = None  # type: ignore


PHOSPHO_SERVER_PATH = str(
    Path(__file__).parent
    / "phospho-mcp-server"
    / "server.py"
)

def get_server_params() -> StdioServerParameters:
    """获取 MCP 服务器参数"""
    default_command = os.environ.get("MCP_COMMAND")
    default_args_env = os.environ.get("MCP_ARGS")
    
    if default_command:
        command = default_command
        args = default_args_env.split() if default_args_env else ["run", PHOSPHO_SERVER_PATH]
    else:
        command = "mcp"
        args = ["run", PHOSPHO_SERVER_PATH]

    return StdioServerParameters(
        command=command,
        args=args,
        env=None,
        cwd=str(Path(__file__).parent),
    )


def get_llm_client() -> Any:
    if ZhipuAiClient is None:
        raise RuntimeError("请先执行: pip install zai-sdk")
    api_key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 ZHIPUAI_API_KEY (智谱 AI 的 API Key)")
    return ZhipuAiClient(api_key=api_key)


def build_tools_brief(tools_obj: Any) -> str:
    """构建工具列表的简要描述"""
    lines = []
    for t in getattr(tools_obj, "tools", []) or []:
        name = getattr(t, "name", "")
        desc = getattr(t, "description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)

async def run_chat(model: str | None = None, debug: bool = False, thinking: bool = False) -> None:
    """AI 机械蛇宠物主程序 - 持续对话模式"""
    model = model or os.environ.get("OPENAI_MODEL", "glm-4.5v")
    server_params = get_server_params()
    llm = get_llm_client()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tools_brief = build_tools_brief(tools)

            # 创建 AI 代理
            ai_agent = AIAgent(llm, model=model, debug=debug, enable_thinking=thinking)
            ai_agent.set_system_prompt(tools_brief)
            
            if thinking:
                print("⚠️  已启用thinking模式，可能影响响应格式")
            else:
                print("✅ 使用标准模式，确保JSON格式输出")

            print("🐍 AI 机械蛇宠物已上线！")
            print("输入指令开始对话，输入 'q'、'quit' 或 'exit' 退出。")
            print("-" * 50)

            while True:
                try:
                    user_message = input("\n你: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n再见！👋")
                    break
                
                if user_message.lower() in {"q", "quit", "exit"}:
                    print("再见！👋")
                    break
                    
                if not user_message:
                    continue

                print("\n助手正在思考并执行...")
                
                # 创建任务执行器并执行任务
                task_executor = TaskExecutor(session, ai_agent, max_steps=10, debug=debug, enable_camera=True)
                
                try:
                    result = await task_executor.execute_task(user_message, initial_image=True)
                    print(f"\n🐍 助手: {result}")
                    
                    if debug:
                        summary = task_executor.get_execution_summary()
                        print(f"\n[DEBUG] 执行摘要: {summary}")
                        
                except Exception as e:
                    print(f"\n❌ 执行失败: {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                
                print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 机械蛇宠物 - 智能对话控制")
    parser.add_argument("--model", default=None, help="LLM 模型名（默认从 OPENAI_MODEL 读取或 glm-4.5v）")
    parser.add_argument("--debug", action="store_true", help="开启调试日志")
    parser.add_argument("--thinking", action="store_true", help="启用智谱AI的thinking模式（可能影响JSON格式输出）")
    args = parser.parse_args()

    # 启动 AI 机械蛇宠物
    asyncio.run(run_chat(model=args.model, debug=args.debug, thinking=args.thinking))