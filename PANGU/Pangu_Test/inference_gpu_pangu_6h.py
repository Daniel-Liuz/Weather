import os
import numpy as np
import onnx
import onnxruntime as ort
from datetime import datetime, timedelta

# --- 1. 用户配置区 ---

# --- 1.1 路径配置 ---
# ONNX 模型文件路径
onnx_model_path = r"E:\Project_Collection\2025\WEATHER\PANGU\pre_trained_model\pangu_weather_6.onnx"

# 基础项目目录 (用于存放处理后的NPY数据和模型输出)
base_project_dir = r"E:\Project_Collection\2025\WEATHER\PANGU\DATA\6h\npy"

# 输入的 .npy 数据目录 (预处理阶段生成的)
input_npy_dir_surface = os.path.join(base_project_dir, "processed_npy_data", "surface_data_6h")
input_npy_dir_upper = os.path.join(base_project_dir, "processed_npy_data", "upper_data_6h")

# 模型预测结果 (.npy 文件) 的输出目录
output_npy_dir_model = os.path.join(base_project_dir, "model_output_6h")

# --- 1.2 预测参数配置 ---
# 初始时间场的时间戳字符串 (格式: YYYYMMDDTHH)
initial_time_str = "20250604T18"  # <--- 修改为您实际的起始时间

# 预测的6小时步数 (例如, 4 步 = 预测未来 24 小时)
run_steps = 1  # <--- 修改为您希望预测的步数

# --- 1.3 文件名前缀 (通常保持不变) ---
prefix_surface_vars = "mslp-10mU-10mV-2mT"
prefix_upper_vars = "z-q-t-u-v"


# --- 2. 脚本执行区 ---

def main():
    print("--- Pangu-Weather 6小时模型 GPU 推理脚本 ---")
    print(f"模型路径: {onnx_model_path}")
    print(f"输入地表数据NPY目录: {input_npy_dir_surface}")
    print(f"输入高空数据NPY目录: {input_npy_dir_upper}")
    print(f"输出NPY目录: {output_npy_dir_model}")
    print(f"初始时间: {initial_time_str}")
    print(f"预测步数: {run_steps} (即 {run_steps * 6} 小时)")

    # 创建输出目录 (如果不存在)
    os.makedirs(output_npy_dir_model, exist_ok=True)
    print(f"确保输出目录 '{output_npy_dir_model}' 已创建或已存在。")

    # --- 2.1 加载 ONNX 模型并初始化 ONNX Runtime 会话 ---
    try:
        print(f"\n正在加载 ONNX 模型: {onnx_model_path} ...")
        # onnx.load() 只是为了检查模型是否可读，实际推理使用 InferenceSession 加载
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)  # 可选：检查模型有效性
        print("ONNX 模型加载并检查通过。")
    except FileNotFoundError:
        print(f"错误：ONNX 模型文件在路径 '{onnx_model_path}' 未找到。请检查路径。")
        return
    except Exception as e:
        print(f"错误：加载或检查 ONNX 模型失败: {e}")
        return

    # 设置 ONNX Runtime 会话选项
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 可以尝试开启以获取潜在性能提升
    options.intra_op_num_threads = 1  # 对于GPU推理，通常设置为1

    # CUDA provider 选项
    cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}

    try:
        print("正在初始化 ONNX Runtime 推理会话 (尝试使用 CUDA)...")
        ort_session = ort.InferenceSession(
            onnx_model_path,
            sess_options=options,
            providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
        )
        if 'CUDAExecutionProvider' in ort_session.get_providers():
            print("ONNX Runtime 会话成功初始化并使用 CUDAExecutionProvider。")
        else:
            print("警告：CUDAExecutionProvider 不可用或未被选中，将使用 CPUExecutionProvider。")
    except Exception as e:
        print(f"错误：初始化 ONNX Runtime 推理会话失败: {e}")
        print("请确保已正确安装 onnxruntime-gpu 并配置好 CUDA 环境，或者模型与当前环境兼容。")
        return

    # --- 2.2 加载初始场的输入数据 ---
    initial_file_name_upper = f"{prefix_upper_vars}-{initial_time_str}.npy"
    initial_file_name_surface = f"{prefix_surface_vars}-{initial_time_str}.npy"

    initial_upper_path = os.path.join(input_npy_dir_upper, initial_file_name_upper)
    initial_surface_path = os.path.join(input_npy_dir_surface, initial_file_name_surface)

    try:
        print(f"\n正在加载初始高空数据: {initial_upper_path}")
        if not os.path.exists(initial_upper_path):
            print(f"错误：Python 确认高空文件不存在于: {initial_upper_path}")
            # 尝试列出该目录下的内容，帮助排查
            try:
                print(f"目录 '{input_npy_dir_upper}' 下的文件列表: {os.listdir(input_npy_dir_upper)}")
            except FileNotFoundError:
                print(f"错误：高空数据目录 '{input_npy_dir_upper}' 本身也不存在。")
            except Exception as e_list_upper:
                print(f"错误：尝试列出高空数据目录时发生错误: {e_list_upper}")
            return  # 因为文件不存在，直接退出函数
        else:
            print("Python 确认高空文件存在。尝试加载...")
            try:
                current_input_upper = np.load(initial_upper_path).astype(np.float32)
                print(f"初始高空数据形状: {current_input_upper.shape}")
            except Exception as e_load_upper:
                print(f"错误：加载高空文件时发生错误: {e_load_upper}")
                return
        print(f"正在加载初始地表数据: {initial_surface_path}")
        if not os.path.exists(initial_surface_path):
            print(f"错误：Python 确认地表文件不存在于: {initial_surface_path}")
            # 尝试列出该目录下的内容，帮助排查
            try:
                print(f"目录 '{input_npy_dir_surface}' 下的文件列表: {os.listdir(input_npy_dir_surface)}")
            except FileNotFoundError:
                print(f"错误：地表数据目录 '{input_npy_dir_surface}' 本身也不存在。")
            except Exception as e_list_surface:
                print(f"错误：尝试列出地表数据目录时发生错误: {e_list_surface}")
            return  # 因为文件不存在，直接退出函数
        else:
            print("Python 确认地表文件存在。尝试加载...")
            try:
                current_input_surface = np.load(initial_surface_path).astype(np.float32)
                print(f"初始地表数据形状: {current_input_surface.shape}")
            except Exception as e_load_surface:
                print(f"错误：加载地表文件时发生错误: {e_load_surface}")
                return
    except FileNotFoundError:
        print(f"错误：初始输入文件未找到。请检查路径和文件名是否正确，以及 '{initial_time_str}' 是否有对应的预处理数据。")
        print(f"  尝试查找高空数据: {initial_upper_path}")
        print(f"  尝试查找地表数据: {initial_surface_path}")
        return
    except Exception as e:
        print(f"错误：加载初始输入数据时发生错误: {e}")
        return

    # --- 2.3 执行迭代预测 ---
    print(f"\n--- 开始迭代预测，共 {run_steps} 步 ---")

    # 将初始时间字符串转换为 datetime 对象，用于计算后续时间
    try:
        current_valid_time_dt = datetime.strptime(initial_time_str, "%Y%m%dT%H")
    except ValueError:
        print(f"错误：初始时间字符串 '{initial_time_str}' 格式不正确。应为 YYYYMMDDTHH。")
        return

    for i in range(run_steps):
        step_num = i + 1
        print(f"\n--- 预测第 {step_num} / {run_steps} 步 ---")
        print(f"当前输入数据对应的有效时间: {current_valid_time_dt.strftime('%Y-%m-%d %H:%M UTC')}")

        # 模型预测的是当前有效时间之后6小时的情况
        forecast_target_time_dt = current_valid_time_dt + timedelta(hours=6)
        forecast_target_time_str = forecast_target_time_dt.strftime("%Y%m%dT%H")
        print(
            f"本步预测的目标时间: {forecast_target_time_dt.strftime('%Y-%m-%d %H:%M UTC')} (文件名时间戳: {forecast_target_time_str})")

        # Pangu模型期望的输入名是 'input' (高空) 和 'input_surface' (地表)
        ort_inputs = {
            'input': current_input_upper,
            'input_surface': current_input_surface
        }

        try:
            # 运行推理
            predicted_upper_all_levels, predicted_surface_all_vars = ort_session.run(None, ort_inputs)
            print(
                f"  推理完成。高空预测形状: {predicted_upper_all_levels.shape}, 地表预测形状: {predicted_surface_all_vars.shape}")

            # 保存当前步骤的预测结果
            output_file_name_upper = f'{prefix_upper_vars}-{forecast_target_time_str}.npy'
            output_file_name_surface = f'{prefix_surface_vars}-{forecast_target_time_str}.npy'

            path_to_save_upper = os.path.join(output_npy_dir_model, output_file_name_upper)
            path_to_save_surface = os.path.join(output_npy_dir_model, output_file_name_surface)

            np.save(path_to_save_upper, predicted_upper_all_levels)
            np.save(path_to_save_surface, predicted_surface_all_vars)
            print(f"  高空预测结果已保存到: {path_to_save_upper}")
            print(f"  地表预测结果已保存到: {path_to_save_surface}")

            # 更新下一次迭代的输入：当前预测成为下一次的输入
            current_input_upper = predicted_upper_all_levels
            current_input_surface = predicted_surface_all_vars

            # 更新当前输入的有效时间，为下一次迭代做准备
            current_valid_time_dt = forecast_target_time_dt

        except Exception as e:
            print(f"错误：在预测第 {step_num} 步时发生错误: {e}")
            print("迭代预测已中止。")
            return

    print("\n--- 所有迭代预测步骤已完成 ---")
    print(f"最终预测结果已保存到目录: {output_npy_dir_model}")


if __name__ == "__main__":
    main()
