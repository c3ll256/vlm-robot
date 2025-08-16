# VLM Robot 架构说明

## 架构概述

重构后的代码采用清晰的分层架构，将AI调用、任务执行和用户交互分离：

```
bot_control.py       (主程序 - 用户交互)
    ↓
ai_agent.py         (AI代理 - LLM交互)
    ↓  
task_executor.py    (任务执行器 - 工具调用和多步推理)
    ↓
MCP Server          (机械臂工具)
```

## 核心组件

### 1. AIAgent (ai_agent.py)
- **职责**: 与LLM交互，理解用户指令，决定下一步行动
- **输入**: 用户指令、图像、上下文信息
- **输出**: 行动决策 (tool/observe/complete/think/error)
- **特点**: 
  - 支持多模态输入（文本+图像）
  - 结构化的JSON响应解析
  - 对话历史管理

### 2. TaskExecutor (task_executor.py) 
- **职责**: 执行具体工具调用，管理多步推理流程
- **功能**:
  - 循环式多步推理（最多10步）
  - 自动获取摄像头图像
  - 工具执行结果反馈
  - 执行状态跟踪
- **策略**: 
  - 移动后自动获取新图像
  - 智能图像更新检测
  - 异常处理和错误恢复

### 3. BotControl (bot_control.py)
- **职责**: 主程序，处理用户交互和程序入口
- **功能**: 持续对话模式，循环接收用户指令直到退出

## 执行流程

### 持续对话流程
```
启动程序 → 
  循环 {
    等待用户输入 →
    多步推理执行 {
      AIAgent决策 → 
      if (tool) TaskExecutor执行工具 → 获取新图像 →
      if (observe) 获取新图像 →
      if (think) 继续思考 →
      if (complete) 返回结果并继续等待下一个指令
    } 直到任务完成或达到最大步数
  } 直到用户退出
```

## 使用方式

```bash
# 启动持续对话（默认）
python bot_control.py

# 开启调试模式
python bot_control.py --debug

# 指定特定模型
python bot_control.py --model "gpt-4" --debug
```

## 优势

1. **职责分离**: AI逻辑与工具执行分离，便于维护和测试
2. **可扩展性**: 新增工具或AI能力无需修改核心流程
3. **鲁棒性**: 完善的错误处理和状态管理
4. **调试友好**: 详细的调试日志和执行摘要
5. **多模态**: 天然支持文本+图像的多模态交互

## 配置

- `ZHIPUAI_API_KEY`: 智谱 AI API 密钥（兼容 `OPENAI_API_KEY`）
- `OPENAI_MODEL`: 模型名称（默认: glm-4.5v）
- `MCP_COMMAND`: MCP命令（可选）
- `MCP_ARGS`: MCP参数（可选）

## 智谱 AI SDK 配置

系统已切换为智谱 AI 官方 SDK，支持更好的多模态功能：

### 安装依赖
```bas
pip install zai-sdk
```

### 环境变量配置
```bash
export ZHIPUAI_API_KEY="your_zhipuai_api_key_here"
```

### 支持的模型
- `glm-4.5v`: 新旗舰视觉推理模型，支持图像、视频、文档理解
- `glm-4.1v-thinking`: 思考模式模型

参考文档：[智谱 AI GLM-4.5V](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.5v#python)
