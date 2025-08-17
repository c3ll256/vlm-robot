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
            # 构建API调用参数
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 8192
            }
            
            # 根据配置决定是否启用thinking模式
            if self.enable_thinking:
                api_params["thinking"] = {"type": "enabled"}
                self._debug_log("启用thinking模式")
            else:
                self._debug_log("禁用thinking模式")
            
            response = self.llm.chat.completions.create(**api_params)
            
            # 调试：输出完整的响应结构
            self._debug_log("=== 完整的 response 结构 ===")
            self._debug_log(f"response 类型: {type(response)}")
            self._debug_log(f"response 属性: {dir(response) if hasattr(response, '__dict__') else 'No attributes'}")
            
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
            
            # 检查可能的响应位置
            possible_content_locations = []
            if hasattr(response, 'choices') and response.choices:
                for i, choice in enumerate(response.choices):
                    if hasattr(choice, 'message'):
                        if hasattr(choice.message, 'content'):
                            possible_content_locations.append(f"choices[{i}].message.content: {choice.message.content}")
                        if hasattr(choice.message, 'reasoning_content'):
                            possible_content_locations.append(f"choices[{i}].message.reasoning_content: {choice.message.reasoning_content}")
                    if hasattr(choice, 'text'):
                        possible_content_locations.append(f"choices[{i}].text: {choice.text}")
                    if hasattr(choice, 'content'):
                        possible_content_locations.append(f"choices[{i}].content: {choice.content}")
            
            if hasattr(response, 'content'):
                possible_content_locations.append(f"response.content: {response.content}")
            if hasattr(response, 'text'):
                possible_content_locations.append(f"response.text: {response.text}")
            
            self._debug_log("可能的内容位置:")
            for location in possible_content_locations:
                self._debug_log(f"  {location}")
            self._debug_log("=== 响应结构分析结束 ===")
            
            # 智谱AI在thinking模式下，内容可能在reasoning_content中
            content = ""
            if response.choices:
                message = response.choices[0].message
                # 首先尝试从content获取
                if hasattr(message, 'content') and message.content and message.content.strip():
                    content = message.content
                # 如果content为空或只是换行，尝试从reasoning_content获取
                elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                    # 检查reasoning_content是否包含JSON格式
                    reasoning_text = message.reasoning_content
                    if '<|begin_of_box|>' in reasoning_text and '<|end_of_box|>' in reasoning_text:
                        content = reasoning_text
                        self._debug_log("从reasoning_content获取JSON内容")
                    else:
                        # reasoning_content是自然语言，不是JSON，记录但不使用
                        self._debug_log("reasoning_content包含自然语言而非JSON，跳过")
                        self._debug_log(f"reasoning_content preview: {reasoning_text[:200]}...")
            
            self._debug_log("最终提取的 content:", content)
            
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
                        
                        # 构建重试API参数
                        retry_params = {
                            "model": self.model,
                            "messages": retry_messages,
                            "temperature": 0.2,
                            "max_tokens": 8192
                        }
                        
                        if self.enable_thinking:
                            retry_params["thinking"] = {"type": "enabled"}
                        
                        retry_response = self.llm.chat.completions.create(**retry_params)
                        
                        # 调试：重试响应的结构
                        self._debug_log("=== 重试 response 结构 ===")
                        self._debug_log(f"retry_response 类型: {type(retry_response)}")
                        
                        # 检查重试响应的可能内容位置
                        retry_possible_content = []
                        if hasattr(retry_response, 'choices') and retry_response.choices:
                            for i, choice in enumerate(retry_response.choices):
                                if hasattr(choice, 'message'):
                                    if hasattr(choice.message, 'content'):
                                        retry_possible_content.append(f"choices[{i}].message.content: {choice.message.content}")
                                    if hasattr(choice.message, 'reasoning_content'):
                                        retry_possible_content.append(f"choices[{i}].message.reasoning_content: {choice.message.reasoning_content}")
                                if hasattr(choice, 'text'):
                                    retry_possible_content.append(f"choices[{i}].text: {choice.text}")
                                if hasattr(choice, 'content'):
                                    retry_possible_content.append(f"choices[{i}].content: {choice.content}")
                        
                        self._debug_log("重试响应的可能内容位置:")
                        for location in retry_possible_content:
                            self._debug_log(f"  {location}")
                        self._debug_log("=== 重试响应结构分析结束 ===")
                        
                        # 重试时也检查reasoning_content
                        retry_content = ""
                        if retry_response.choices:
                            retry_message = retry_response.choices[0].message
                            # 首先尝试从content获取
                            if hasattr(retry_message, 'content') and retry_message.content and retry_message.content.strip():
                                retry_content = retry_message.content
                            # 如果content为空，尝试从reasoning_content获取
                            elif hasattr(retry_message, 'reasoning_content') and retry_message.reasoning_content:
                                retry_reasoning = retry_message.reasoning_content
                                if '<|begin_of_box|>' in retry_reasoning and '<|end_of_box|>' in retry_reasoning:
                                    retry_content = retry_reasoning
                                    self._debug_log("重试：从reasoning_content获取JSON内容")
                                else:
                                    self._debug_log("重试：reasoning_content包含自然语言而非JSON，跳过")
                        
                        self._debug_log("重试最终提取的 content:", retry_content)
                        
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
