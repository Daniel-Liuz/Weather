import os
import glob
import numpy as np

# --- 配置区 ---
# 这个路径指向你存放所有npy数据的DATA文件夹
BASE_DATA_DIR = r"/PANGU/Pangu_Prediction\DATA"
# 掩码文件路径 (我们之前生成的，并已移动到正确位置)
CHINA_MASK_PATH = r"/PANGU/Pangu_Prediction\agent\masks\china_mask.npy"


# --- 工具函数 ---
def get_average_temperature_for_china(time_interval: str, step: int) -> str:
    """
    获取未来指定时间点中国的平均2米温度。

    这个函数是Agent可以调用的"工具"。它执行实际的计算。

    :param time_interval: 预报间隔，字符串格式，如 '1h', '3h', '6h'。
    :param step: 预测的步数，整数，从1开始。例如，对于'1h'间隔，step=1表示1小时后。
    :return: 一个包含计算结果或错误信息的字符串，供Agent理解。
    """
    print(f"\n--- 工具被调用 ---")
    print(f"参数: time_interval='{time_interval}', step={step}")

    try:
        # 1. 验证并定位数据文件夹
        data_dir = os.path.join(BASE_DATA_DIR, f"animation_data_{time_interval}")
        if not os.path.isdir(data_dir):
            error_msg = f"错误: 找不到 '{time_interval}' 模式的数据目录: {data_dir}"
            print(error_msg)
            return error_msg

        # 2. 找到对应的npy文件
        search_pattern = os.path.join(data_dir, 'mslp-*.npy')
        npy_files = sorted(glob.glob(search_pattern))

        if not npy_files:
            error_msg = f"错误: 在目录 {data_dir} 中没有找到任何 'mslp-*.npy' 文件。"
            print(error_msg)
            return error_msg

        # 检查 step 是否在有效范围内 (1 到 文件总数)
        if not (1 <= step <= len(npy_files)):
            error_msg = f"错误: step={step} 超出范围。'{time_interval}' 模式只有 {len(npy_files)} 步预测数据 (从1到{len(npy_files)})。"
            print(error_msg)
            return error_msg

        # 列表索引是从0开始的，所以用 step-1
        target_file_path = npy_files[step - 1]
        print(f"目标文件: {os.path.basename(target_file_path)}")

        # 3. 加载数据和掩码
        surface_data = np.load(target_file_path)
        t2m_k = surface_data[3, :, :]  # 提取第4个变量: 2米温度 (开尔文)

        if not os.path.exists(CHINA_MASK_PATH):
            error_msg = f"致命错误: 找不到中国区域的掩码文件 {CHINA_MASK_PATH}。"
            print(error_msg)
            return error_msg
        china_mask = np.load(CHINA_MASK_PATH)

        # 4. 计算平均温度
        # 使用掩码过滤数据，只保留中国区域的温度值
        # where china_mask is 1, keep the temperature, otherwise it's ignored
        china_temperatures_k = t2m_k[china_mask == 1]

        if china_temperatures_k.size == 0:
            error_msg = "错误: 掩码应用后没有找到任何数据点。检查掩码文件是否正确。"
            print(error_msg)
            return error_msg

        avg_temp_k = np.mean(china_temperatures_k)
        avg_temp_c = avg_temp_k - 273.15

        success_msg = f"计算完成：基于 '{time_interval}' 模型的第 {step} 步预测，中国的平均2米温度约为 {avg_temp_c:.2f} 摄氏度。"
        print(success_msg)
        return success_msg

    except Exception as e:
        error_msg = f"执行工具时发生未知错误: {e}"
        print(error_msg)
        return error_msg


# --- 用于独立测试本脚本的区域 ---
if __name__ == "__main__":
    print("=" * 40)
    print("开始独立测试 weather_tools.py")
    print("=" * 40)

    # 测试用例 1: 正常情况
    print("\n[测试 1] 查询 6h 模型第 2 步 (未来12小时) 的数据")
    result1 = get_average_temperature_for_china(time_interval='6h', step=2)

    # 测试用例 2: 边界条件测试 (查询 1h 模型的最后一步)
    print("\n[测试 2] 查询 1h 模型第 24 步 (未来24小时) 的数据")
    result2 = get_average_temperature_for_china(time_interval='1h', step=24)

    # 测试用例 3: 错误情况 (step 超出范围)
    print("\n[测试 3] 查询 6h 模型第 5 步 (不存在)")
    result3 = get_average_temperature_for_china(time_interval='6h', step=5)

    # 测试用例 4: 错误情况 (模式名称错误)
    print("\n[测试 4] 查询一个不存在的 '5h' 模型")
    result4 = get_average_temperature_for_china(time_interval='5h', step=1)

    print("\n" + "=" * 40)
    print("测试结束。")
    print("=" * 40)
