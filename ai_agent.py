"""
AI Agent 类 - 专门负责与 LLM 交互和理解指令
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple


class AIAgent:
    """AI 代理类，负责与 LLM 交互，理解用户指令并生成执行计划"""
    
    def __init__(self, llm_client: Any, model: str = "glm-4.5v", debug: bool = False):
        self.llm = llm_client
        self.model = model
        self.debug = debug
        self.system_prompt = ""
        self.conversation_history: List[Dict[str, Any]] = []
        
    def set_system_prompt(self, tools_brief: str):
        """设置系统提示，包含工具列表"""
        self.system_prompt = (
            "你是一条有温度、有好奇心的 AI 机械蛇宠物，既会说话也会行动。\n"
            "你的任务是理解用户指令并决定下一步行动。\n\n"
            "⚠️ 重要：必须严格按照JSON格式输出，不要添加任何其他文字！\n\n"
            "输出规则（只输出JSON，不要其他内容）：\n"
            "1. 如果需要执行工具操作，输出：{\"action\": \"tool\", \"tool\": \"工具名\", \"arguments\": {参数对象}, \"reasoning\": \"执行原因\"}\n"
            "2. 如果需要更多信息或观察，输出：{\"action\": \"observe\", \"reasoning\": \"需要观察的原因\"}\n"
            "3. 如果任务完成，输出：{\"action\": \"complete\", \"response\": \"最终回复\", \"reasoning\": \"完成原因\"}\n"
            "4. 如果需要思考下一步，输出：{\"action\": \"think\", \"reasoning\": \"思考内容\"}\n\n"
            "执行策略：\n"
            "- 涉及面向人的社交动作时，先根据摄像头判断对方方位并调整朝向\n"
            "- 朝向调整：画面左侧→rz取正角度，右侧→rz取负角度，每次10~30度\n"
            "- 确认对齐后再执行社交动作，未对齐前不要完成任务\n"
            "- 工具执行后观察新状态，确认效果\n\n"
            f"可用工具：\n{tools_brief}\n\n"
            "记住：只输出纯JSON格式，不要包含任何其他文字或标记！"
        )
    
    def _debug_log(self, *args):
        """调试日志输出"""
        if self.debug:
            print("[AI_AGENT_DEBUG]", *args)
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON"""
        if not content:
            return None
            
        content = content.strip()
        
        # 去掉常见包装标记
        for token in ("<|begin_of_box|>", "<|end_of_box|>"):
            content = content.replace(token, "")
        
        # 去除代码块标记
        content = re.sub(r"^```(?:json)?\n([\s\S]*?)\n```$", r"\1", content, flags=re.IGNORECASE)
        
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 提取第一个 JSON 对象
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
        
        return None
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
    
    def add_to_history(self, role: str, content: Any):
        """添加到对话历史"""
        self.conversation_history.append({"role": role, "content": content})
        
        # 限制对话历史长度，避免超过token限制
        max_history_length = 20  # 最多保留20轮对话
        if len(self.conversation_history) > max_history_length:
            # 保留最近的对话，移除最早的
            self.conversation_history = self.conversation_history[-max_history_length:]
            self._debug_log(f"对话历史已裁剪至最近 {max_history_length} 轮")
    
    def decide_next_action(self, user_input: str, image_data_url: Optional[str] = None, 
                          context: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        决定下一步行动
        
        Returns:
            Tuple[str, Dict[str, Any]]: (action_type, action_data)
            action_type: "tool", "observe", "complete", "think", "error"
        """
        
        # 构建用户输入内容
        user_content = []
        if isinstance(user_input, str) and user_input.strip():
            user_content.append({"type": "text", "text": user_input})
        
        if image_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        
        # 添加上下文信息
        if context:
            user_content.append({"type": "text", "text": f"\n当前状态: {context}"})
        
        # 确保至少有一个内容项
        if not user_content:
            user_content.append({"type": "text", "text": "继续执行任务"})
        
        # 构建消息
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_content})
        
        # 调试信息
        self._debug_log(f"消息数量: {len(messages)}")
        self._debug_log(f"用户内容项数: {len(user_content)}")
        self._debug_log(f"包含图像: {any(item.get('type') == 'image_url' for item in user_content)}")
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
                thinking={"type": "enabled"}  # 启用智谱AI的思考模式
            )
            
            content = response.choices[0].message.content if response.choices else ""
            self._debug_log("LLM 原始响应:", content)
            
            # 检查是否为空响应
            if not content or content.strip() == "":
                self._debug_log("LLM 返回空响应，尝试纯文本重试")
                
                # 如果包含图像，尝试纯文本模式重试
                if any(item.get('type') == 'image_url' for item in user_content):
                    self._debug_log("检测到图像消息，尝试纯文本重试")
                    # 构建纯文本消息
                    text_content = []
                    for item in user_content:
                        if item.get('type') == 'text':
                            text_content.append(item)
                    
                    if not text_content:
                        text_content = [{"type": "text", "text": "继续执行任务，已获取摄像头图像"}]
                    
                    # 构建纯文本消息重试
                    retry_messages = [{"role": "system", "content": self.system_prompt}]
                    retry_messages.extend(self.conversation_history[-5:])  # 只取最近5轮历史
                    retry_messages.append({"role": "user", "content": text_content})
                    
                    try:
                        self._debug_log("尝试纯文本模式重试")
                        retry_response = self.llm.chat.completions.create(
                            model=self.model,
                            messages=retry_messages,
                            temperature=0.2,
                            max_tokens=1000,
                            thinking={"type": "enabled"}
                        )
                        retry_content = retry_response.choices[0].message.content if retry_response.choices else ""
                        if retry_content and retry_content.strip():
                            self._debug_log("纯文本重试成功")
                            content = retry_content
                        else:
                            self._debug_log("纯文本重试也失败")
                            return "error", {"error": "LLM 返回空响应，可能需要重新开始任务"}
                    except Exception as e:
                        self._debug_log(f"纯文本重试异常: {e}")
                        return "error", {"error": "LLM 返回空响应，可能需要重新开始任务"}
                else:
                    # 非图像消息的空响应处理
                    if len(self.conversation_history) > 5:
                        self.conversation_history = self.conversation_history[-5:]
                        self._debug_log("已清理对话历史，建议重新开始任务")
                    return "error", {"error": "LLM 返回空响应，可能需要重新开始任务"}
            
            # 解析响应
            parsed = self._parse_json_response(content)
            if not parsed:
                self._debug_log("JSON 解析失败，原始内容:", content)
                return "error", {"error": "无法解析 LLM 响应", "raw_content": content}
            
            action = parsed.get("action", "").lower()
            if action not in ["tool", "observe", "complete", "think"]:
                return "error", {"error": f"未知的动作类型: {action}", "parsed": parsed}
            
            # 添加到对话历史
            self.add_to_history("assistant", content)
            
            return action, parsed
            
        except Exception as e:
            self._debug_log("LLM 调用异常:", str(e))
            return "error", {"error": str(e)}
    
    def process_tool_result(self, tool_name: str, tool_result: Any, success: bool = True):
        """处理工具执行结果，添加到对话历史"""
        if success:
            # 简化工具结果，避免过长的内容
            if isinstance(tool_result, dict):
                # 只保留关键信息
                simplified_result = {}
                if "content" in tool_result:
                    content = tool_result["content"]
                    if isinstance(content, list) and content:
                        # 只取第一个内容项的摘要
                        first_content = content[0]
                        if hasattr(first_content, 'text'):
                            text = str(first_content.text)
                            # 限制文本长度
                            simplified_result["message"] = text[:100] + "..." if len(text) > 100 else text
                        else:
                            simplified_result["message"] = "命令已执行"
                    else:
                        simplified_result["message"] = "命令已执行"
                else:
                    simplified_result["message"] = "命令已执行"
                
                result_text = f"工具 {tool_name} 执行成功: {simplified_result['message']}"
            else:
                # 其他类型的结果
                result_str = str(tool_result)
                if len(result_str) > 100:
                    result_str = result_str[:100] + "..."
                result_text = f"工具 {tool_name} 执行成功: {result_str}"
        else:
            result_text = f"工具 {tool_name} 执行失败: {tool_result}"
        
        self.add_to_history("user", result_text)
        self._debug_log("工具结果已添加到历史:", result_text)
    
    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        if not self.conversation_history:
            return "无对话历史"
        
        return f"对话轮次: {len(self.conversation_history)}, 最后一轮: {self.conversation_history[-1]['role']}"
