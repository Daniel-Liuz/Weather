# 文件名: 1_ResNet_vs_Persistence.py (保存在 pages/ 目录下)

import streamlit as st
from PIL import Image
import os

# --- 核心修改：使用健壮的路径定位方法 ---
# 获取当前脚本文件所在的目录 (即 'pages' 目录)
PAGES_DIR = os.path.dirname(__file__)
# 获取项目的根目录 ('AI_Weather_Platform')
ROOT_DIR = os.path.join(PAGES_DIR, os.pardir)
# 构建到可视化文件的绝对路径
VIS_DIR = os.path.join(ROOT_DIR, "data_visuals", "resnet_climatelearn")


def main():
    st.set_page_config(page_title="ResNet对比分析", page_icon="🌪️", layout="wide")

    st.title("🌪️ ResNet vs. Persistence | 效果对比")
    st.markdown(
        "这是一个交互式应用，用于对比我们训练的 **ResNet AI模型** 与 **持续性（Persistence）基准模型** 的预测效果。")

    # 定义需要展示的文件
    gif_files = {
        "resnet_temp": os.path.join(VIS_DIR, "resnet_temperature.gif"),
        "persist_temp": os.path.join(VIS_DIR, "persistence_temperature.gif"),
        "resnet_geo": os.path.join(VIS_DIR, "resnet_geopotential.gif"),
        "persist_geo": os.path.join(VIS_DIR, "persistence_geopotential.gif"),
    }

    # (其余代码与你原来的版本完全相同，这里为了简洁省略)
    # (你可以直接复制你原来的代码，只需要替换掉最上面的路径定义部分即可)

    # 检查文件是否存在
    for key, path in gif_files.items():
        if not os.path.exists(path):
            st.error(f"错误：找不到文件 '{os.path.basename(path)}'！请确认它在以下目录中：\n{VIS_DIR}")
            return

    tab1, tab2 = st.tabs(["🌡️ 温度 (Temperature) 对比", "📈 位势高度 (Geopotential) 对比"])
    with tab1:
        st.header("🌡️ 24小时温度预测对比")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ ResNet AI 模型预测")
            st.image(gif_files["resnet_temp"], caption="ResNet模型对温度的预测结果")
        with col2:
            st.subheader("☑️ Persistence 基准模型预测")
            st.image(gif_files["persist_temp"], caption="Persistence模型对温度的预测结果")
    with tab2:
        st.header("📈 24小时位势高度预测对比")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ ResNet AI 模型预测")
            st.image(gif_files["resnet_geo"], caption="ResNet模型对位势高度的预测结果")
        with col2:
            st.subheader("☑️ Persistence 基准模型预测")
            st.image(gif_files["persist_geo"], caption="Persistence模型对位势高度的预测结果")


if __name__ == "__main__":
    main()
