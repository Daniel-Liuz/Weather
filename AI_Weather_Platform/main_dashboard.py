import streamlit as st

st.set_page_config(
    page_title="AI气象大模型综合展示平台",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 欢迎来到我的AI气象大模型综合展示平台")
st.markdown(
    """
    ### 关于本项目
    本项目集成了多个前沿的AI气象预测模型，并提供了可视化的预测结果分析。
    您可以通过左侧的导航栏切换不同的模型应用。

    **当前集成的应用包括:**
    - **ResNet vs. Persistence**: 基于 `climate-learn` 框架，对比了基础的ResNet模型与Persistence基准模型的预测效果。
    - **Pangu Forecast**: 展示了华为盘古气象大模型的预测结果，代表了当前领域的最先进水平。
    - **Weather Agent**: 提供基于最优集成策略的温度预测，支持6h、12h、24h和48h的预测。
    """
)
