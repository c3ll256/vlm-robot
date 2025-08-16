"""
任务执行器 - 负责执行具体的工具调用和管理多步推理流程
"""
import asyncio
import time
import hashlib
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
from mcp import ClientSession
from ai_agent import AIAgent


class TaskExecutor:
    """任务执行器，负责管理多步推理和工具调用"""
    
    def __init__(self, session: ClientSession, ai_agent: AIAgent, max_steps: int = 10, debug: bool = False):
        self.session = session
        self.ai_agent = ai_agent
        self.max_steps = max_steps
        self.debug = debug
        self.executed_tools: Set[str] = set()
        self.step_count = 0
        self.last_image_hash: Optional[str] = None
        
    def _debug_log(self, *args):
        """调试日志输出"""
        if self.debug:
            print("[TASK_EXECUTOR_DEBUG]", *args)
    
    def _load_phosphobot_client(self):
        """动态加载 PhosphoClient 类"""
        phosphobot_path = Path(__file__).parent / "phospho-mcp-server" / "tools" / "phosphobot.py"
        
        spec = importlib.util.spec_from_file_location("phosphobot", phosphobot_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 phosphobot 模块: {phosphobot_path}")
        
        phosphobot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(phosphobot_module)
        
        return phosphobot_module.PhosphoClient
    
    def _fetch_camera_frame(self, force_new: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        获取摄像头帧
        
        Returns:
            Tuple[Optional[str], Optional[str]]: (data_url, content_hash)
        """
        try:
            PhosphoClient = self._load_phosphobot_client()
            client = PhosphoClient()
            result = client.get("/frames", params={"_t": int(time.time() * 1000)})
            
            if isinstance(result, dict) and result:
                # 取第一个帧
                first_value = next(iter(result.values()))
                if isinstance(first_value, str) and first_value:
                    content_hash = hashlib.md5(first_value.encode()).hexdigest()
                    
                    # 如果不强制获取新帧且哈希相同，返回 None
                    if not force_new and self.last_image_hash == content_hash:
                        return None, content_hash
                    
                    data_url = f"data:image/jpeg;base64,{first_value}"
                    self.last_image_hash = content_hash
                    return data_url, content_hash
                    
        except Exception as e:
            self._debug_log(f"获取摄像头帧失败: {e}")
        
        return None, None
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        执行单个工具
        
        Returns:
            Tuple[bool, Any]: (success, result)
        """
        try:
            self._debug_log(f"执行工具: {tool_name}, 参数: {arguments}")
            result = await self.session.call_tool(name=tool_name, arguments=arguments)
            self.executed_tools.add(tool_name.lower())
            return True, result
        except Exception as e:
            self._debug_log(f"工具执行失败: {tool_name}, 错误: {e}")
            return False, str(e)
    
    def _build_context_info(self) -> str:
        """构建当前上下文信息"""
        context_parts = [
            f"步骤: {self.step_count}/{self.max_steps}",
            f"已执行工具: {', '.join(sorted(self.executed_tools)) if self.executed_tools else '无'}",
        ]
        return " | ".join(context_parts)
    
    async def execute_task(self, user_input: str, initial_image: bool = True) -> str:
        """
        执行完整任务，支持多步推理
        
        Args:
            user_input: 用户输入的指令
            initial_image: 是否在开始时获取图像
            
        Returns:
            str: 最终回复
        """
        self._debug_log(f"开始执行任务: {user_input}")
        
        # 重置状态
        self.step_count = 0
        self.executed_tools.clear()
        self.ai_agent.reset_conversation()
        
        # 获取初始图像
        current_image_url = None
        if initial_image:
            current_image_url, _ = self._fetch_camera_frame(force_new=True)
        
        # 开始多步推理循环
        while self.step_count < self.max_steps:
            self.step_count += 1
            self._debug_log(f"=== 步骤 {self.step_count} ===")
            
            # 构建上下文
            context = self._build_context_info()
            
            # AI 决定下一步行动
            action_type, action_data = self.ai_agent.decide_next_action(
                user_input=user_input if self.step_count == 1 else "",  # 只在第一步传入用户输入
                image_data_url=current_image_url,
                context=context
            )
            
            self._debug_log(f"AI 决策: {action_type}, 数据: {action_data}")
            
            if action_type == "error":
                error_msg = action_data.get("error", "未知错误")
                return f"执行出错: {error_msg}"
            
            elif action_type == "complete":
                # 任务完成
                response = action_data.get("response", "任务已完成")
                self._debug_log(f"任务完成: {response}")
                return response
            
            elif action_type == "think":
                # AI 需要思考，继续下一轮
                reasoning = action_data.get("reasoning", "思考中...")
                self._debug_log(f"AI 思考: {reasoning}")
                continue
            
            elif action_type == "observe":
                # 需要观察，获取新的图像
                self._debug_log("获取新的观察图像")
                new_image_url, _ = self._fetch_camera_frame(force_new=True)
                if new_image_url:
                    current_image_url = new_image_url
                    self._debug_log("已获取新的摄像头图像")
                else:
                    self._debug_log("无法获取摄像头图像")
                continue
            
            elif action_type == "tool":
                # 执行工具
                tool_name = action_data.get("tool")
                arguments = action_data.get("arguments", {})
                reasoning = action_data.get("reasoning", "")
                
                if not tool_name:
                    self.ai_agent.process_tool_result("", "工具名称缺失", success=False)
                    continue
                
                # 执行工具
                success, result = await self._execute_tool(tool_name, arguments)
                
                # 将结果反馈给 AI
                self.ai_agent.process_tool_result(tool_name, result, success=success)
                
                # 如果是移动类工具，获取新图像
                if tool_name.lower() in ["move_relative", "move_absolute", "move_init"]:
                    self._debug_log("移动后获取新图像")
                    # 等待一小段时间让机械臂稳定
                    await asyncio.sleep(0.5)
                    new_image_url, _ = self._fetch_camera_frame(force_new=True)
                    if new_image_url:
                        current_image_url = new_image_url
                        self._debug_log("已获取移动后的摄像头图像")
                
                continue
            
            else:
                # 未知动作类型
                self._debug_log(f"未知动作类型: {action_type}")
                return f"AI 返回了未知的动作类型: {action_type}"
        
        # 达到最大步数限制
        return f"任务执行超过最大步数限制 ({self.max_steps})，可能需要更具体的指令。"
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "total_steps": self.step_count,
            "executed_tools": list(self.executed_tools),
            "conversation_summary": self.ai_agent.get_conversation_summary()
        }
