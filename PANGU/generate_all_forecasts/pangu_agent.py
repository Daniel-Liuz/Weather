import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
# 1. 导入新的HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
import os
import sys

# --- 将agent目录添加到Python的搜索路径中 ---
agent_dir = os.path.join(os.path.dirname(__file__), 'agent')
sys.path.append(agent_dir)
from weather_tools import get_average_temperature_for_china

# --- 配置和常量 ---
MODEL_PATH = r"E:\Project_Collection\2025\WEATHER\model\1.5B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- 定义Agent可以使用的工具 ---
@tool
def china_average_temperature_tool(time_interval: str, step: int) -> str:
    """
    当用户想要查询未来某个特定时间点'中国'的'平均温度'时，使用此工具。
    这个工具需要两个参数：
    - 'time_interval': 字符串类型，表示预测的时间间隔，必须是 '1h', '3h', '6h' 或 '24h' 中的一个。
    - 'step': 整数类型，表示预测的步数，从1开始。
    例如，要查询6小时模型的第2步（即未来12小时）的数据，参数应为 time_interval='6h', step=2。
    """
    return get_average_temperature_for_china(time_interval, step)


# --- 缓存资源加载函数 (非常重要) ---
@st.cache_resource
def load_llm_and_tokenizer():
    """加载本地LLM和Tokenizer。"""
    st.info(f"开始加载模型: {MODEL_PATH} 到 {DEVICE}...")
    try:
        # 2. 保持强制使用CUDA
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE  # 直接使用我们定义的DEVICE变量
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # 3. 创建LLM对象的方式发生改变
        # 我们不再创建一个transformers的pipeline，而是直接将模型和tokenizer交给LangChain
        # 这给了我们更精细的控制
        llm = HuggingFacePipeline.from_model_id(
            model_id=MODEL_PATH,
            model_kwargs={"torch_dtype": torch.bfloat16},
            # device_map=DEVICE 会自动被处理，确保模型在GPU上
            # HuggingFacePipeline 会更智能地处理设备放置
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 1024,
                "do_sample": False,
                "temperature": 0.1,
                "top_p": 0.95,
            },
        )
        st.success("模型和Tokenizer加载成功！")
        return llm
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        st.stop()


# --- 创建Agent (这部分代码保持不变) ---
def create_pangu_agent(llm):
    tools = [china_average_temperature_tool]
# 这是我们的新template，对Action Input提出了更严格的要求
    template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: The input for the action. IMPORTANT: This MUST be a single, valid JSON object. All keys and all string values in the JSON MUST be enclosed in double quotes. Do not add any text before or after the JSON object.
Example: {{"time_interval": "6h", "step": 1}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor


# --- Streamlit 应用界面 (这部分代码保持不变) ---
st.set_page_config(page_title="盘古气象智能助手", page_icon="🧠")
st.title("盘古气象智能助手 💬")
st.caption("一个能与盘古模型数据对话的AI Agent")

llm = load_llm_and_tokenizer()
agent_executor = create_pangu_agent(llm)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "你好！我是盘古气象智能助手。你可以问我关于未来中国平均温度的问题，例如：'未来6小时后中国的平均温度是多少？'"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 智能助手正在思考并计算..."):
            response = agent_executor.invoke({"input": prompt})
            final_answer = response['output']
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

