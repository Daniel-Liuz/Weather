# 1_download_data.py

import climate_learn as cl
import os
import torch

# --- 本地环境配置 ---
# 我们把原来Google Drive的路径 "/content/drive/MyDrive/..."
# 替换成一个我们本地的路径。
# os.getcwd() 会获取当前脚本所在的文件夹（即CLIMATE_LEARN_PROJECT）
# 然后我们指定所有数据都存放在其下的一个新建的 "data" 文件夹里。
LOCAL_DATA_ROOT = os.path.join(os.getcwd(), "data")
def main():
    """
    主函数，负责执行数据下载任务。
    """
    print("ClimateLearn快速入门 - 第1步：下载数据")
    print("=======================================")

    # 检查GPU是否可用，为之后的步骤做准备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到可用设备: {device}")

    # --- 下载温度数据 ---
    # 定义存放温度数据的具体路径
    temperature_dst = os.path.join(LOCAL_DATA_ROOT, "temperature")
    print(f"\n准备下载 'temperature_850' 数据到: {temperature_dst}")
    # os.makedirs 会自动创建这个文件夹（如果它不存在的话）
    os.makedirs(temperature_dst, exist_ok=True)

    # 调用ClimateLearn核心的下载函数
    # 我们把'dst'参数改成了我们本地的路径
    cl.data.download_weatherbench(
        dst=temperature_dst,
        dataset="era5",
        variable="temperature_850"
    )
    print("温度数据下载完成。")

    # --- 下载位势高度数据 ---
    # 定义存放位势高度数据的具体路径
    geopotential_dst = os.path.join(LOCAL_DATA_ROOT, "geopotential")
    print(f"\n准备下载 'geopotential_500' 数据到: {geopotential_dst}")
    os.makedirs(geopotential_dst, exist_ok=True)

    # 调用ClimateLearn核心的下载函数
    cl.data.download_weatherbench(
        dst=geopotential_dst,
        dataset="era5",
        variable="geopotential_500"
    )
    print("位势高度数据下载完成。")

    print("\n=======================================")
    print("数据下载阶段成功结束！")
    print(f"请检查 '{LOCAL_DATA_ROOT}' 文件夹，确认数据文件已下载。")


# 这是一个标准的Python写法，确保main()函数只在直接运行此脚本时执行
if __name__ == "__main__":
    main()

