import os
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta

# --- 1. Master Configuration ---

# <<<--- 您只需要在这里修改要运行的模式 ---<<<
MODEL_TYPE_TO_RUN = '3h'  # 可选值: '1h', '3h', '6h', '24h'

# 初始时间点 (您最新的数据)
INITIAL_TIME_STR = "20250604T18"

# 定义所有模型的配置
MODEL_CONFIGS = {
    '1h': {
        'model_path': r"E:\Project_Collection\2025\WEATHER\PANGU\pre_trained_model\pangu_weather_1.onnx",
        'run_steps': 24,  # 运行24步，覆盖24小时
        'time_delta_hours': 1,
        'output_subdir': 'animation_data_1h'
    },
    '3h': {
        'model_path': r"E:\Project_Collection\2025\WEATHER\PANGU\pre_trained_model\pangu_weather_3.onnx",
        'run_steps': 8,  # 运行8步，覆盖24小时
        'time_delta_hours': 3,
        'output_subdir': 'animation_data_3h'
    },
    '6h': {
        'model_path': r"E:\Project_Collection\2025\WEATHER\PANGU\pre_trained_model\pangu_weather_6.onnx",
        'run_steps': 4,  # 运行4步，覆盖24小时
        'time_delta_hours': 6,
        'output_subdir': 'animation_data_6h'
    },
    '24h': {
        'model_path': r"E:\Project_Collection\2025\WEATHER\PANGU\pre_trained_model\pangu_weather_24.onnx",
        'run_steps': 1,  # 只运行1步，得到24小时后的结果
        'time_delta_hours': 24,
        'output_subdir': 'animation_data_24h'
    }
}


# --- Don't touch below unless you know what you are doing ---

def main():
    # --- 2. Setup based on selected configuration ---
    config = MODEL_CONFIGS[MODEL_TYPE_TO_RUN]
    onnx_model_path = config['model_path']
    run_steps = config['run_steps']
    time_delta_hours = config['time_delta_hours']

    print(f"--- Pangu-Weather Forecast Generation Script ---")
    print(f"模式: {MODEL_TYPE_TO_RUN}")
    print(f"模型路径: {onnx_model_path}")
    print(f"初始时间: {INITIAL_TIME_STR}")
    print(f"预测步数: {run_steps} (每步 {time_delta_hours} 小时, 共 {run_steps * time_delta_hours} 小时)")

    # --- 3. Path Configuration (Updated as per your request) ---

    # --- Input Paths (where the initial data comes from) ---
    input_base_dir = r"E:\Project_Collection\2025\WEATHER\PANGU\DATA\6h\npy"
    input_npy_dir_surface = os.path.join(input_base_dir, "processed_npy_data", "surface_data_6h")
    input_npy_dir_upper = os.path.join(input_base_dir, "processed_npy_data", "upper_data_6h")

    # --- Output Paths (where the generated forecasts will be saved) ---
    # 这是您项目的主文件夹
    output_project_dir = r"/PANGU/Pangu_Prediction"
    # 我们将在这个主文件夹下创建一个 'DATA' 目录来存放所有输出
    output_npy_dir_model = os.path.join(output_project_dir, "DATA", config['output_subdir'])

    # 确保最终的输出目录存在
    os.makedirs(output_npy_dir_model, exist_ok=True)

    print(f"输入地表数据来自: {input_npy_dir_surface}")
    print(f"输入高空数据来自: {input_npy_dir_upper}")
    print(f"输出将保存到新路径: {output_npy_dir_model}")

    # --- 4. Initialization and Data Loading (same as before) ---
    try:
        ort_session = ort.InferenceSession(
            onnx_model_path,
            providers=[('CUDAExecutionProvider', {}), 'CPUExecutionProvider']
        )
        if 'CUDAExecutionProvider' not in ort_session.get_providers():
            print("警告: CUDA 不可用，将使用 CPU。")
        else:
            print("ONNX Runtime 会话成功初始化并使用 CUDA。")
    except Exception as e:
        print(f"错误: 初始化 ONNX Runtime 会话失败: {e}")
        return

    prefix_surface_vars = "mslp-10mU-10mV-2mT"
    prefix_upper_vars = "z-q-t-u-v"

    initial_file_surface = os.path.join(input_npy_dir_surface, f"{prefix_surface_vars}-{INITIAL_TIME_STR}.npy")
    initial_file_upper = os.path.join(input_npy_dir_upper, f"{prefix_upper_vars}-{INITIAL_TIME_STR}.npy")

    try:
        current_input_surface = np.load(initial_file_surface).astype(np.float32)
        current_input_upper = np.load(initial_file_upper).astype(np.float32)
        print("初始场数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 找不到初始场文件 for {INITIAL_TIME_STR}。请确保文件存在。")
        print(f"  - {initial_file_surface}")
        print(f"  - {initial_file_upper}")
        return

    # --- 5. Iterative Prediction (same as before) ---
    print(f"\n--- 开始迭代预测 ---")
    current_valid_time_dt = datetime.strptime(INITIAL_TIME_STR, "%Y%m%dT%H")

    for i in range(run_steps):
        step_num = i + 1
        forecast_target_time_dt = current_valid_time_dt + timedelta(hours=time_delta_hours)
        forecast_target_time_str = forecast_target_time_dt.strftime("%Y%m%dT%H")

        print(f"--- 步骤 {step_num}/{run_steps}: 预测 {forecast_target_time_str} ---")

        ort_inputs = {'input': current_input_upper, 'input_surface': current_input_surface}
        predicted_upper, predicted_surface = ort_session.run(None, ort_inputs)

        output_file_surface = os.path.join(output_npy_dir_model,
                                           f"{prefix_surface_vars}-{forecast_target_time_str}.npy")
        output_file_upper = os.path.join(output_npy_dir_model, f"{prefix_upper_vars}-{forecast_target_time_str}.npy")

        np.save(output_file_surface, predicted_surface)
        np.save(output_file_upper, predicted_upper)
        print(f"  结果已保存。")

        current_input_surface, current_input_upper = predicted_surface, predicted_upper
        current_valid_time_dt = forecast_target_time_dt

    print("\n--- 所有预测步骤已完成 ---")


if __name__ == "__main__":
    main()
