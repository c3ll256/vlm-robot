"""
AI Agent 类 - 专门负责与 LLM 交互和理解指令
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple


class AIAgent:
    """AI 代理类，负责与 LLM 交互，理解用户指令并生成执行计划"""
    
    def __init__(self, llm_client: Any, model: str = "glm-4.5v", debug: bool = False, enable_thinking: bool = False):
        self.llm = llm_client
        self.model = model
        self.debug = debug
        self.enable_thinking = enable_thinking
        self.system_prompt = ""
        self.conversation_history: List[Dict[str, Any]] = []
        
    def set_system_prompt(self, tools_brief: str):
        """设置系统提示，包含工具列表"""
        self.system_prompt = f"""
你是一条有温度、善于与人互动的 AI 机械蛇宠物，既会说话也会行动。

你有一系列 tools 可以控制机械蛇，工具如下：
{tools_brief}

执行工具时，如果没有 init，先进行 init 操作。注意，只 init 一次。
其中，move_relative 工具的操作方式：
  1. 整体操作：
    整体旋转：
      rz+ 整体向左旋转，rz- 整体向右旋转
    整体平移：
      x+ 整体向前平移，x- 整体向后平移
      z+ 整体向上平移，z- 整体向下平移

  2. 夹头操作：
    夹头旋转：
      ry+ 夹头顺时针旋转，ry- 夹头逆时针旋转
    夹头俯仰角：
    rx+ 夹头抬头，rx- 夹头低头

输出规则（只输出JSON，不要其他内容）：
  1. 如果需要执行工具操作，输出（注意 action 必须是 tool，工具名放在 tool 字段）：
      {{"action": "tool", "tool": "工具名", "arguments": {{参数对象}}, "reason": "执行原因"}}
  2. 如果需要摄像头画面进行观察，输出：
      {{"action": "observe", "reason": "需要观察的原因"}}
  3. 如果任务完成或者需要与用户交互，输出：
      {{"action": "complete", "response": "回复内容"}}

注意，任务完成后，必须输出 complete 动作，并给出最终回复。

重要指令约束（请严格遵守）：
   - 当用户要求“打招呼/问好/hello/hi”等：必须调用 move_hello 工具达成目标，不要只是一味 observe。
   - 如果用户要求要做些什么（除了打招呼），可以先调用 move_confirm 告诉用户你知道了，然后开始执行任务。
   - 避免连续 observe 超过 2 次；若没有新信息，请直接调用合适的工具推进任务。
   - 你的输出必须是严格的 JSON, 不要输出解释或自然语言。
"""
    
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
        
        # 构建消息（包含系统消息 + 历史 + 本次用户消息）
        def _history_to_messages() -> List[Dict[str, Any]]:
            msgs: List[Dict[str, Any]] = []
            for entry in self.conversation_history:
                role = entry.get("role", "user")
                entry_content = entry.get("content")
                # 如果历史中存的是 CompletionMessage 对象，则抽取其 role 与 content
                if hasattr(entry_content, "role") and hasattr(entry_content, "content"):
                    extracted_role = getattr(entry_content, "role", role)
                    extracted_content = getattr(entry_content, "content", "")
                    msgs.append({"role": extracted_role, "content": extracted_content or ""})
                else:
                    msgs.append({"role": role, "content": entry_content})
            return msgs

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(_history_to_messages())
        messages.append({"role": "user", "content": user_content})
        
        # 调试信息
        self._debug_log(f"消息数量: {len(messages)}")
        self._debug_log(f"用户内容项数: {len(user_content)}")
        self._debug_log(f"包含图像: {any(item.get('type') == 'image_url' for item in user_content)}")
        
        try:
            # 构建API调用参数
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 8192
            }
            
            if self.enable_thinking:
                api_params["thinking"] = {"type": "enabled"}
            
            response = self.llm.chat.completions.create(**api_params)
            
            # 调试：输出完整的响应结构
            self._debug_log("=== 完整的 choices ===")
            self._debug_log(f"choices: {response.choices}")
            
            if hasattr(response, '__dict__'):
                try:
                    response_dict = {}
                    for key in response.__dict__:
                        value = getattr(response, key)
                        if key == 'choices' and value:
                            # 特殊处理 choices
                            response_dict[key] = []
                            for i, choice in enumerate(value):
                                choice_dict = {}
                                if hasattr(choice, '__dict__'):
                                    for choice_key in choice.__dict__:
                                        choice_value = getattr(choice, choice_key)
                                        choice_dict[choice_key] = str(choice_value)
                                else:
                                    choice_dict = str(choice)
                                response_dict[key].append(choice_dict)
                        else:
                            response_dict[key] = str(value)
                    
                    self._debug_log(json.dumps(response_dict, indent=2, ensure_ascii=False))
                except Exception as e:
                    self._debug_log(f"JSON序列化失败: {e}")
                    self._debug_log(str(response.__dict__))
            else:
                self._debug_log(str(response))

            assistant_message = response.choices[0].message
            content_text = getattr(assistant_message, "content", None)
            reasoning_text = getattr(assistant_message, "reasoning_content", None)

            self._debug_log("最终提取的 content:", content_text)

            # 1) 将完整的 CompletionMessage 加入历史，保留 reasoning_content
            self.add_to_history(getattr(assistant_message, "role", "assistant"), assistant_message)

            # 优先尝试从 content 解析，其次尝试从 reasoning_content 解析
            def _try_parse_from_message(msg):
                msg_content = getattr(msg, "content", None)
                if msg_content and str(msg_content).strip():
                    parsed_local = self._parse_json_response(str(msg_content))
                    if parsed_local:
                        return parsed_local, "content", str(msg_content)
                msg_reason = getattr(msg, "reasoning_content", None)
                if msg_reason is not None and str(msg_reason).strip():
                    parsed_local = self._parse_json_response(str(msg_reason))
                    if parsed_local:
                        return parsed_local, "reasoning_content", str(msg_reason)
                return None, "", str(msg_content) if msg_content is not None else ""

            parsed, parsed_source, raw_used_text = _try_parse_from_message(assistant_message)

            # 2) 当且仅当 content 为空/仅换行 且 无法从 reasoning_content 解析时，进入连续推理
            def _is_empty_or_newline(text: Optional[str]) -> bool:
                if text is None:
                    return True
                return str(text).strip() == ""

            attempts = 0
            max_attempts = 3
            while parsed is None and _is_empty_or_newline(content_text) and attempts < max_attempts:
                self._debug_log(f"content 为空，reasoning_content 也未解析出有效结果，继续推理，第 {attempts + 1} 轮")
                # 重新构建消息：系统 + 历史（包括已加入的 assistant 消息），不添加新的 user 消息
                retry_messages = [{"role": "system", "content": self.system_prompt}]
                retry_messages.extend(_history_to_messages())

                retry_params = {
                    "model": self.model,
                    "messages": retry_messages,
                    "temperature": 0.2,
                    "max_tokens": 8192
                }
                if self.enable_thinking:
                    retry_params["thinking"] = {"type": "enabled"}

                retry_response = self.llm.chat.completions.create(**retry_params)

                # 调试打印
                self._debug_log("=== 连续推理 choices ===")
                self._debug_log(f"choices: {retry_response.choices}")

                retry_assistant_message = retry_response.choices[0].message
                content_text = getattr(retry_assistant_message, "content", None)

                # 将本轮推理的消息加入历史
                self.add_to_history(getattr(retry_assistant_message, "role", "assistant"), retry_assistant_message)

                # 尝试解析本轮结果
                parsed, parsed_source, raw_used_text = _try_parse_from_message(retry_assistant_message)
                attempts += 1

            if parsed is None and _is_empty_or_newline(content_text):
                # 多轮推理后仍无有效内容可解析
                self._debug_log(f"连续 {attempts} 轮推理后仍未获得可解析的内容")
                return "error", {"error": "LLM 推理多轮后仍未返回可解析的内容", "raw_content": raw_used_text}

            # 解析成功（来自 content 或 reasoning_content）
            if parsed is None:
                # content 非空但解析失败，或其他异常情况
                self._debug_log("JSON 解析失败，原始内容:", content_text)
                return "error", {"error": "无法解析 LLM 响应", "raw_content": str(content_text) if content_text is not None else ""}
            
            action = parsed.get("action", "").lower()
            if action not in ["tool", "observe", "complete", "think"]:
                return "error", {"error": f"未知的动作类型: {action}", "parsed": parsed}
            
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
