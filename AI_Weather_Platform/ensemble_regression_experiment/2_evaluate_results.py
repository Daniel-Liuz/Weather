import numpy as np
import os
import matplotlib.pyplot as plt


def calculate_mse(pred_file, true_file, t2m_index=3):
    """计算预测值与真实值之间的均方误差 (MSE)"""
    pred_data = np.load(pred_file)
    true_data = np.load(true_file)
    pred_t2m = pred_data[t2m_index, :, :]
    true_t2m = true_data[t2m_index, :, :]
    mse = np.mean((pred_t2m - true_t2m) ** 2)
    return mse


# 定义预测时长和对应的目录、文件
base_dir = r"E:\Project_Collection\2025\WEATHER\AI_Weather_Platform\ensemble_regression_experiment\regression_dataset"
forecast_configs = {
    6: {
        "dir": os.path.join(base_dir, "T+6h"),
        "true_file": "ground_truth_surface.npy",
        "pred_files": {
            "6 steps of 1h": "pred_surface_from_1h.npy",
            "2 steps of 3h": "pred_surface_from_3h.npy",
            "1 step of 6h": "pred_surface_from_6h.npy"
        }
    },
    12: {
        "dir": os.path.join(base_dir, "T+12h"),
        "true_file": "ground_truth_surface.npy",
        "pred_files": {
            "12 steps of 1h": "pred_surface_from_1h.npy",
            "4 steps of 3h": "pred_surface_from_3h.npy",
            "2 steps of 6h": "pred_surface_from_6h.npy"
        }
    },
    24: {
        "dir": os.path.join(base_dir, "T+24h"),
        "true_file": "ground_truth_surface.npy",
        "pred_files": {
            "24 steps of 1h": "pred_surface_from_1h.npy",
            "8 steps of 3h": "pred_surface_from_3h.npy",
            "4 steps of 6h": "pred_surface_from_6h.npy",
            "1 step of 24h": "pred_surface_from_24h.npy"
        }
    },
    48: {
        "dir": os.path.join(base_dir, "T+48h"),
        "true_file": "ground_truth_surface.npy",
        "pred_files": {
            "48 steps of 1h": "pred_surface_from_1h.npy",
            "16 steps of 3h": "pred_surface_from_3h.npy",
            "8 steps of 6h": "pred_surface_from_6h.npy",
            "2 steps of 24h": "pred_surface_from_24h.npy"
        }
    }
}

# 评估每个预测时长
for ft, config in forecast_configs.items():
    dir_path = config["dir"]
    true_file = os.path.join(dir_path, config["true_file"])
    print(f"\n评估 T+{ft}h 的不同模型组合：")
    min_mse = float('inf')
    best_model = None

    for model_desc, pred_filename in config["pred_files"].items():
        pred_file = os.path.join(dir_path, pred_filename)
        try:
            mse = calculate_mse(pred_file, true_file, t2m_index=3)
            print(f"  {model_desc}: MSE = {mse}")
            if mse < min_mse:
                min_mse = mse
                best_model = model_desc
        except Exception as e:
            print(f"  {model_desc} 失败: {e}")

    print(f"T+{ft}h 最佳模型组合: {best_model}，MSE = {min_mse}")

print("\n评估完成！")

# 可视化每个预测时长的 MSE 对比
for ft, config in forecast_configs.items():
    mses = [calculate_mse(os.path.join(config["dir"], pf), os.path.join(config["dir"], config["true_file"]))
            for pf in config["pred_files"].values()]
    plt.bar(config["pred_files"].keys(), mses)
    plt.title(f"T+{ft}h MSE Comparison")
    plt.xticks(rotation=45)
    plt.show()