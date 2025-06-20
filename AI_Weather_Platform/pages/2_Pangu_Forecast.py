# 文件名: 2_Pangu_Forecast.py (保存在 pages/ 目录下)

import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="盘古模型动态预报", page_icon="🌀", layout="wide")

# --- 核心修改：使用健壮的路径定位方法 ---
PAGES_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(PAGES_DIR, os.pardir)
ANIMATIONS_DIR = os.path.join(ROOT_DIR, "data_visuals", "pangu_weather")

# (从这里开始，代码与你的原版基本一致，只是BASE_DIR相关的部分被替换)
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

st.title("🌀 盘古气象大模型 - 动态预报可视化平台")
st.sidebar.header("动画选择")

selected_plot_name = st.sidebar.radio("选择预报类型:", list(PLOT_TYPE_MAP.keys()))
selected_interval_name = st.sidebar.selectbox("选择预报间隔:", list(INTERVAL_MAP.keys()))

plot_type_key = PLOT_TYPE_MAP[selected_plot_name]
interval_key = INTERVAL_MAP[selected_interval_name]

gif_filename = f"animation_{interval_key}_{plot_type_key}.gif"
gif_path = os.path.join(ANIMATIONS_DIR, gif_filename)

st.header(f"当前显示: {selected_plot_name} - {selected_interval_name}")

if os.path.exists(gif_path):
    st.image(
        gif_path,
        caption=f"预报动画: {gif_filename}",
        use_container_width=True  # <--- 在这里使用新的参数名
    )

    with open(gif_path, "rb") as file:
        st.download_button(
            label=f"下载 {gif_filename}",
            data=file,
            file_name=gif_filename,
            mime="image/gif"
        )
else:
    st.error(f"错误: 动画文件 '{gif_filename}' 未找到！")
    st.warning(f"请确认它在以下目录中生成: \n{ANIMATIONS_DIR}")