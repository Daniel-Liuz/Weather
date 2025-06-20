import os
import numpy as np
import onnxruntime as ort
import shutil
import gc

# 配置区
MODEL_DIR = r"E:\Project_Collection\2025\WEATHER\PANGU\pre_trained_model"
INITIAL_DATA_DIR = r"E:\Project_Collection\2025\WEATHER\PANGU\DATA\6h\npy\processed_npy_data"
OUTPUT_DATASET_DIR = os.path.join(os.path.dirname(__file__), "regression_dataset")
START_TIME_STR = "20250601T00"
LEAD_TIMES_TO_GENERATE = [48, 72]

# 辅助函数
def load_model(step):
    model_name = f"pangu_weather_{step}.onnx"
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = True
    providers = [
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider',
    ]
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

def run_pangu_step(input_surface, input_upper, ort_session):
    ort_inputs = {
        'input': input_upper.astype(np.float32),
        'input_surface': input_surface.astype(np.float32)
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    del ort_inputs
    gc.collect()
    return ort_outputs[1], ort_outputs[0]

def generate_prediction_path(initial_surface, initial_upper, total_hours, step_size):
    current_surface, current_upper = initial_surface.copy(), initial_upper.copy()
    num_steps = total_hours // step_size
    ort_session = load_model(step_size)  # 按需加载模型

    print(f"    Generating path for T+{total_hours}h using {num_steps} steps of {step_size}h model...")
    for i in range(num_steps):
        print(f"      - Step {i + 1}/{num_steps}...")
        current_surface, current_upper = run_pangu_step(current_surface, current_upper, ort_session)
    del ort_session  # 释放模型
    gc.collect()
    return current_surface, current_upper

if __name__ == "__main__":
    print("--- 可用的 ONNX Runtime 执行提供程序 ---")
    print(ort.get_available_providers())
    print("---------------------------------------")

    # 加载初始状态 (T=0)
    start_surface_path = os.path.join(INITIAL_DATA_DIR, 'surface_data_6h', f'mslp-10mU-10mV-2mT-{START_TIME_STR}.npy')
    start_upper_path = os.path.join(INITIAL_DATA_DIR, 'upper_data_6h', f'z-q-t-u-v-{START_TIME_STR}.npy')
    try:
        initial_surface = np.load(start_surface_path)
        initial_upper = np.load(start_upper_path)
        print(f"\n成功加载初始场数据: {START_TIME_STR}")
    except FileNotFoundError:
        print(f"[致命错误] 找不到初始场文件...")
        exit()

    # 循环为每个定义好的预报时效生成数据
    for lead_time in LEAD_TIMES_TO_GENERATE:
        print(f"\n===== 生成回归数据集: T+{lead_time}h =====")
        target_dir = os.path.join(OUTPUT_DATASET_DIR, f"T+{lead_time}h")
        os.makedirs(target_dir, exist_ok=True)
        possible_steps = [6, 24]
        for step in possible_steps:
            if lead_time % step == 0:
                pred_surface_full, _ = generate_prediction_path(initial_surface, initial_upper, lead_time, step)
                save_path = os.path.join(target_dir, f"pred_surface_from_{step}h.npy")
                np.save(save_path, pred_surface_full)
                print(f"      -> 已保存预测结果: {os.path.basename(save_path)}")
                gc.collect()

        # 准备标准答案
        start_time_formatted = f"{START_TIME_STR[:4]}-{START_TIME_STR[4:6]}-{START_TIME_STR[6:8]} {START_TIME_STR[9:11]}:00"
        gt_hour_dt = np.datetime64(start_time_formatted) + np.timedelta64(lead_time, 'h')
        gt_hour_str = np.datetime_as_string(gt_hour_dt, unit='h').replace('-', '').replace(':', '')
        gt_filename = f'mslp-10mU-10mV-2mT-{gt_hour_str}.npy'
        gt_source_path = os.path.join(INITIAL_DATA_DIR, 'surface_data_6h', gt_filename)
        gt_dest_path = os.path.join(target_dir, 'ground_truth_surface.npy')

        if os.path.exists(gt_source_path):
            shutil.copy(gt_source_path, gt_dest_path)
            print(f"      -> 已复制标准答案: {os.path.basename(gt_source_path)}")
        else:
            print(f"[错误] 标准答案文件未找到: {gt_source_path}")

    print("\n\n数据集生成任务全部完成！")
