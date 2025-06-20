# 文件名: 3_run_baselines.py

import climate_learn as cl
import pytorch_lightning as pl
import os
import torch


def main():
    """
    主函数，负责定义数据加载器、加载基准模型并进行测试。
    """
    print("ClimateLearn快速入门 - 阶段3：建立并测试基准模型")
    print("===================================================")

    # 检查GPU是否可用。这一步虽然计算量不大，但习惯性检查一下。
    # 真正的模型训练将非常需要GPU。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到可用设备: {device}\n")

    # --- 1. 定义数据加载器 (DataModule) ---
    print("正在设置数据加载器 (DataModule)...")

    # 获取我们处理好的 .npz 数据所在的路径
    processed_data_dir = os.path.join(os.getcwd(), "data", "processed")
    print(f"将从这个路径加载数据: {processed_data_dir}")

    # 这是项目的核心配置之一，它告诉模型我们想做什么
    dm = cl.data.IterDataModule(
        task="direct-forecasting",  # 任务类型：直接预测
        inp_root_dir=processed_data_dir,  # 输入数据来自我们处理好的文件夹
        out_root_dir=processed_data_dir,  # 输出数据也来自那里（因为我们要预测未来的自己）
        in_vars=["temperature", "geopotential"],  # 输入变量
        out_vars=["temperature", "geopotential"],  # 输出变量
        src="era5",
        subsample=6,  # 数据采样频率：每6小时一次
        pred_range=24,  # 预测范围：预测未来24小时后的情况
        history=3,  # 历史数据长度：用过去3个时间点的数据(t, t-6h, t-12h)来做预测
        batch_size=32  # 批处理大小：每次给模型32个样本
    )
    dm.setup()  # 执行设置，准备好数据管道
    print("数据加载器设置完成。\n")

    # --- 2. 加载并测试“气候学”基准模型 ---
    print("--- 开始测试 Climatology (气候学) 模型 ---")
    climatology = cl.load_forecasting_module(
        data_module=dm,
        architecture="climatology"  # 指定加载“气候学”模型
    )

    # 创建一个 PyTorch Lightning 的“教练” (Trainer)
    # 它会自动处理测试流程
    trainer = pl.Trainer(accelerator=device)
    trainer.test(climatology, dm)
    print("--- Climatology 模型测试完成 ---\n")

    # --- 3. 加载并测试“持续性”基准模型 ---
    print("--- 开始测试 Persistence (持续性) 模型 ---")
    persistence = cl.load_forecasting_module(
        data_module=dm,
        architecture="persistence"  # 指定加载“持续性”模型
    )

    trainer.test(persistence, dm)
    print("--- Persistence 模型测试完成 ---\n")

    print("===================================================")
    print("阶段3成功结束！你现在已经有了两个基准模型的性能分数。")
    print("下一步，我们将训练一个真正的深度学习模型，并挑战超越这些分数！")


if __name__ == "__main__":
    main()
