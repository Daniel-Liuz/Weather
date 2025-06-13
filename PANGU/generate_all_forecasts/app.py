import streamlit as st
import os
from PIL import Image

# --- 1. Page Configuration ---
# 设置页面标题、图标和布局
st.set_page_config(
    page_title="盘古模型动态预报",
    page_icon="🌀",
    layout="wide"
)

# --- 2. Define Paths and Mappings ---
# 基础目录
BASE_DIR = os.path.dirname(__file__)
ANIMATIONS_DIR = os.path.join(BASE_DIR, 'ANIMATIONS')

# 创建一个字典，将用户友好的显示名称映射到文件名中的关键字
# 这样可以轻松地调整显示文本，而无需更改代码逻辑
PLOT_TYPE_MAP = {
    "天气系统 (等压线+风场)": "weather_system",
    "2米温度分布": "temperature"
}

INTERVAL_MAP = {
    "1小时模型 (24帧)": "1h",
    "3小时模型 (8帧)": "3h",
    "6小时模型 (4帧)": "6h",
    "24小时模型 (1帧)": "24h"
}

# --- 3. App Layout and Content ---

# --- Header ---
st.title("🌀 盘古气象大模型 - 动态预报可视化平台")
st.markdown(
    """
    这是一个基于 Streamlit 的交互式平台，用于展示由盘古气象大模型生成的不同类型和时间间隔的天气预报动画。
    请使用左侧的侧边栏选择您希望查看的预报类型和模型。
    """
)

# --- Sidebar for User Controls ---
st.sidebar.header("动画选择")

# 用户选择预报类型 (使用 radio buttons)
selected_plot_name = st.sidebar.radio(
    "选择预报类型:",
    list(PLOT_TYPE_MAP.keys())
)

# 用户选择预报间隔 (使用 select box)
selected_interval_name = st.sidebar.selectbox(
    "选择预报间隔:",
    list(INTERVAL_MAP.keys())
)

# --- 4. Logic to Display the Selected GIF ---

# 根据用户的选择，从字典中获取对应的文件名关键字
plot_type_key = PLOT_TYPE_MAP[selected_plot_name]
interval_key = INTERVAL_MAP[selected_interval_name]

# 构建要显示的GIF文件名和完整路径
gif_filename = f"animation_{interval_key}_{plot_type_key}.gif"
gif_path = os.path.join(ANIMATIONS_DIR, gif_filename)

# 在主页面显示一个标题，告诉用户当前正在查看什么
st.header(f"当前显示: {selected_plot_name} - {selected_interval_name}")

# 检查GIF文件是否存在
if os.path.exists(gif_path):
    # 如果文件存在，则显示GIF
    st.image(gif_path, caption=f"预报动画: {gif_filename}", use_column_width=True)

    # 提供一个下载按钮
    with open(gif_path, "rb") as file:
        st.download_button(
            label=f"下载 {gif_filename}",
            data=file,
            file_name=gif_filename,
            mime="image/gif"
        )
else:
    # 如果文件不存在，显示一个友好的错误提示
    st.error(f"错误: 动画文件未找到！")
    st.warning(f"请确认文件 '{gif_filename}' 是否已在以下目录中生成: \n{ANIMATIONS_DIR}")
    st.info("如果文件缺失，请运行 `create_animations.py` 脚本来生成所有动画。")

# --- Footer ---
st.markdown("---")
st.markdown("项目由 Pangu-Weather 模型驱动，并使用 Streamlit 进行可视化。")

