# 文件名: 4_train_model.py (最终优化版)

import climate_learn as cl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import os
import torch

# --- 性能优化：根据你的GPU型号和指南建议，开启Tensor Core加速 ---
torch.set_float32_matmul_precision('high')


def main():
    """
    主函数，负责训练和评估ResNet深度学习模型。
    """
    print("ClimateLearn快速入门 - 阶段4：训练并测试 ResNet 模型 (最终优化版)")
    print("==============================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到可用设备: {device}。训练过程将在此设备上进行。\n")

    # --- 1. 定义数据加载器 (最终优化版) ---
    print("正在设置数据加载器...")
    processed_data_dir = os.path.join(os.getcwd(), "data", "processed")

    # ######################################################################
    # # 核心修改点：
    # # 将 num_workers 设置为一个安全值（如4），以避免内存溢出。
    # # 这是解决 OSError 的关键。
    # ######################################################################
    num_workers = 8
    print(f"数据加载将使用 {num_workers} 个 'workers' (CPU核心) 来运行。")

    dm = cl.data.IterDataModule(
        task="direct-forecasting",
        inp_root_dir=processed_data_dir,
        out_root_dir=processed_data_dir,
        in_vars=["temperature", "geopotential"],
        out_vars=["temperature", "geopotential"],
        src="era5",
        subsample=6,
        pred_range=24,
        history=3,
        batch_size=128,
        num_workers=num_workers  # <--- 保留这个最重要的优化！
        # 'persistent_workers' 参数已被移除，因为它不被支持
    )
    dm.setup()
    print("数据加载器设置完成。\n")

    # --- 2. 加载指南中指定的 ResNet 深度学习模型 ---
    print("正在加载 'rasp-theurey-2020' (ResNet) 模型架构...")
    model = cl.load_forecasting_module(
        data_module=dm,
        architecture="rasp-theurey-2020"
    )
    print("模型加载完成。\n")

    # --- 3. 设置高级训练回调函数 (Callbacks) ---
    print("正在设置训练回调函数...")
    default_root_dir = os.path.join(os.getcwd(), "resnet_forecasting_24hrs")
    early_stopping_monitor = "val/lat_mse:aggregate"
    callbacks = [
        RichProgressBar(),
        EarlyStopping(monitor=early_stopping_monitor, patience=5),
        ModelCheckpoint(
            dirpath=os.path.join(default_root_dir, "checkpoints"),
            monitor=early_stopping_monitor,
            mode="min",
            filename="best_model"
        )
    ]
    print("回调函数设置完成: RichProgressBar, EarlyStopping, ModelCheckpoint\n")

    # --- 4. 设置训练日志记录器 ---
    logger = TensorBoardLogger(save_dir=default_root_dir, name="logs")
    print(f"训练日志将保存在: {logger.log_dir}\n")

    # --- 5. 创建并配置“教练” (Trainer) ---
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        accelerator="gpu",
        devices=[0],
        max_epochs=20,
        precision="16-mixed"
    )

    # --- 6. 开始训练！ ---
    print("=====================================================")
    print("即将开始模型训练... 这将花费较长时间，请耐心等待。")
    print(f"目标: 挑战 Persistence 模型的 'lat_rmse' 分数 (303.30)。")
    print(f"监控指标: {early_stopping_monitor} (越低越好)")
    print("=====================================================")

    trainer.fit(model, datamodule=dm)

    print("\n=====================================================")
    print("模型训练结束！")
    print("=====================================================\n")

    # --- 7. 在测试集上评估我们训练出的最佳模型 ---
    print("开始使用测试集评估已保存的最佳模型...")
    trainer.test(datamodule=dm, ckpt_path="best")

    print("\n=====================================================")
    print("阶段4成功结束！")
    print("请查看上方表格中的 'test/lat_rmse:aggregate' 分数，并与基准模型对比。")


if __name__ == "__main__":
    main()
