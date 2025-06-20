# pangu_agent.py (最终版，采用Tool Calling)

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
import os
import sys

# --- 将agent目录添加到Python的搜索路径中 ---
# 确保可以找到 weather_tools.py
# (假设 pangu_agent.py 和 agent/ 文件夹在同一级目录)
# Pangu_Prediction/
# ├── pangu_agent.py
# └── agent/
#     └── weather_tools.py
current_dir = os.path.dirname(__file__)
agent_dir = os.path.join(current_dir, 'agent')
if agent_dir not in sys.path:
    sys.path.append(agent_dir)
from weather_tools import get_average_temperature_for_china

# --- 配置和常量 ---
MODEL_PATH = r"/model/1.5B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- 1. 工具定义 ---
@tool
def china_average_temperature_tool(time_interval: str, step: int) -> str:
    """
    当用户想要查询未来某个特定时间点'中国'的'平均温度'时，使用此工具。
    这个工具需要两个参数：
    - 'time_interval': 字符串类型，表示预测的时间间隔，必须是 '1h', '3h', '6h' 或 '24h' 中的一个。
    - 'step': 整数类型，表示预测的步数，从1开始。
    例如，要查询6小时模型的第2步（即未来12小时）的数据，参数应为 time_interval='6h', step=2。
    """
    # 调用我们已经写好的API函数
    return get_average_temperature_for_china(time_interval, step)


# --- 2. 资源加载 ---
@st.cache_resource
def load_llm_and_tokenizer():
    """加载本地LLM和Tokenizer。"""
    st.info(f"开始加载模型: {MODEL_PATH} 到 {DEVICE}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # 使用 transformers.pipeline 创建一个基础的文本生成管道
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # 对于工具调用，不需要太长的输出
            do_sample=False,
        )

        # 使用 LangChain 的 HuggingFacePipeline 进行包装
        llm = HuggingFacePipeline(pipeline=pipe)

        st.success("模型和Tokenizer加载成功！")
        return llm
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        st.stop()


# --- 3. 创建Agent (核心改动部分) ---
def create_pangu_agent(llm):
    """
    使用更简单、更可靠的 "Tool Calling" Agent, 放弃复杂的ReAct。
    """
    tools = [china_average_temperature_tool]

    # --- 全新的、为Tool Calling设计的Prompt ---
    # 这个Prompt的任务很简单：让模型输出一个JSON，而不是复杂的ReAct格式文本
    template = """You are a helpful assistant that has access to the following tools. 

Here are the tools you can use:
{tools}

To use a tool, respond with a JSON blob containing a single tool call.
The JSON blob should have a "tool_name" and a "tool_input" key.

Here is an example of a valid JSON blob:
```json
{{
  "tool_name": "china_average_temperature_tool",
  "tool_input": {{
    "time_interval": "6h",
    "step": 1
  }}
}}
If the user's question does not require a tool, answer it directly.

Question: {input}
"""
    prompt = PromptTemplate.from_template(template)

    # --- 使用 `create_tool_calling_agent` 替换 `create_react_agent`
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor保持不变，它会自动适配新的Agent类型
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 保持日志输出，方便调试
        handle_parsing_errors=True,  # 保持错误处理
    )
    return agent_executor


st.set_page_config(page_title="盘古气象智能助手", page_icon="🧠")
st.title("盘古气象智能助手 💬")
st.caption("一个能与盘古模型数据对话的AI Agent (工具调用版)")

llm = load_llm_and_tokenizer()
agent_executor = create_pangu_agent(llm)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "你好！我是盘古气象智能助手。你可以问我关于"
                                             "未来中国平均温度的问题，例如：'未来6小时后中国的平均温度是多少？'"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 智能助手正在思考并调用工具..."):
            response = agent_executor.invoke({"input": prompt})
            final_answer = response.get('output', '抱歉，我无法处理您的问题。')
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
