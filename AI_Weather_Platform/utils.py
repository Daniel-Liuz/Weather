import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import os

MODEL_PATH = r"E:\Project_Collection\2025\WEATHER\model\7B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_llm_and_tokenizer():
    """加载本地LLM（例如DEEPSEEK 7B）和Tokenizer。"""
    st.info(f"开始从 {MODEL_PATH} 加载模型到 {DEVICE}...")
    if not os.path.exists(MODEL_PATH):
        st.error(f"错误：模型路径不存在！请检查路径：{MODEL_PATH}")
        st.stop()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        st.success("模型和Tokenizer加载成功！")
        return llm
    except Exception as e:
        st.error(f"加载模型时发生严重错误: {e}")
        st.stop()
