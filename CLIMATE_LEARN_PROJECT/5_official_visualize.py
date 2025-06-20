# 文件名: 5_official_visualize.py (最终完美版 - GIF 输出)

import climate_learn as cl
import os
import torch
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """
    使用 climate-learn 官方推荐的 visualize_at_index 函数，
    并正确地将返回的动画对象保存为 GIF 文件。
    """
    print("ClimateLearn - 最终阶段：使用官方工具进行可视化")
    print("==============================================")

    # --- 1. 设置数据加载器 ---
    print("正在设置数据加载器...")
    processed_data_dir = os.path.join(os.getcwd(), "data", "processed")
    dm = cl.data.IterDataModule(
        task="direct-forecasting", inp_root_dir=processed_data_dir,
        out_root_dir=processed_data_dir, in_vars=["temperature", "geopotential"],
        out_vars=["temperature", "geopotential"], src="era5", subsample=6,
        pred_range=24, history=3, batch_size=1, num_workers=4
    )
    dm.setup()
    print("数据加载器设置完成。\n")

    # --- 2. 加载模型 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("正在加载已训练的 ResNet 模型...")
    checkpoint_path = os.path.join(
        os.getcwd(), "resnet_forecasting_24hrs", "checkpoints", "best_model-v1.ckpt"
    )
    if not os.path.exists(checkpoint_path):
        print(f"错误：找不到 Checkpoint 文件 '{checkpoint_path}'。")
        return

    resnet_model = cl.load_forecasting_module(
        data_module=dm, architecture="rasp-theurey-2020"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    resnet_model.load_state_dict(checkpoint['state_dict'])
    resnet_model.to(device).eval()
    print("ResNet 模型加载成功。\n")

    print("正在加载基准 (Persistence) 模型...")
    persistence_model = cl.load_forecasting_module(
        data_module=dm, architecture="persistence"
    ).to(device).eval()
    print("基准模型加载成功。\n")

    # --- 3. 使用官方 visualize_at_index 函数绘图并保存为 GIF ---

    denorm_transform = resnet_model.test_target_transforms[0]
    output_dir = os.path.join(os.getcwd(), "resnet_climatelearn")
    os.makedirs(output_dir, exist_ok=True)
    print(f"可视化结果将保存到: {output_dir}\n")

    variables_to_plot = ["temperature", "geopotential"]

    for variable in variables_to_plot:
        print(f"--- 正在为变量 '{variable}' 生成可视化动画 ---")

        # ######################################################################
        # # 最终修正:
        # # 1. 函数返回的是一个动画对象 (ArtistAnimation)
        # # 2. 我们使用 animation.save() 方法将其保存为 GIF
        # # 3. 使用 'pillow' 作为 writer，它通常无需额外安装
        # ######################################################################

        # 为 ResNet 模型绘图
        print("  - 正在处理 ResNet 预测结果...")
        anim_resnet = cl.utils.visualize_at_index(
            resnet_model, dm, index=0, variable=variable, src="era5",
            in_transform=denorm_transform,
            out_transform=denorm_transform
        )
        resnet_path = os.path.join(output_dir, f"resnet_{variable}.gif")
        anim_resnet.save(resnet_path, writer='pillow', fps=1)
        plt.close(anim_resnet._fig)  # 关闭动画底层的图，释放内存

        # 为 Persistence 模型绘图
        print("  - 正在处理 Persistence 预测结果...")
        anim_persistence = cl.utils.visualize_at_index(
            persistence_model, dm, index=0, variable=variable, src="era5",
            in_transform=denorm_transform,
            out_transform=denorm_transform
        )
        persistence_path = os.path.join(output_dir, f"persistence_{variable}.gif")
        anim_persistence.save(persistence_path, writer='pillow', fps=1)
        plt.close(anim_persistence._fig)

        print(f"--- 变量 '{variable}' 的动画已成功保存！ ---\n")

    print("==============================================")
    print("🎉🎉🎉 祝贺你！所有可视化任务已全部完成！ 🎉🎉🎉")
    print(f"请检查 '{output_dir}' 文件夹，你将看到以下 GIF 动画文件:")
    print("  - resnet_temperature.gif")
    print("  - resnet_geopotential.gif")
    print("  - persistence_temperature.gif")
    print("  - persistence_geopotential.gif")
    print("\n我们不仅成功了，还得到了比预期更酷的动态结果！")


if __name__ == "__main__":
    main()

