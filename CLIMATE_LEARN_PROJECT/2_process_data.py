# 文件名: 2_process_data.py

from climate_learn.data.processing.nc2npz import convert_nc2npz
import os


def main():
    """
    主函数，负责执行数据处理任务。
    """
    print("ClimateLearn快速入门 - 阶段2：处理数据")
    print("=======================================")
    print("本脚本将执行以下操作:")
    print("1. 读取 'data' 文件夹中的 .nc 文件。")
    print("2. 将数据切分为：训练集 (1979-2014), 验证集 (2015-2016), 测试集 (2017-2018)。")
    print("3. 将处理后的数据以 .npz 格式保存到 'data/processed' 文件夹中。")
    print("这个过程会消耗一些时间，因为它需要读取所有已下载的数据，请耐心等待...")

    # --- 本地环境路径配置 ---
    # 我们下载的数据存放在 'data' 文件夹下
    # 这是 .nc 文件的根目录
    root_dir = os.path.join(os.getcwd(), "data")

    # 我们希望将处理好的 .npz 文件保存到一个新的 'processed' 子文件夹里
    save_dir = os.path.join(root_dir, "processed")

    # 确保保存的文件夹存在
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n数据源路径 (root_dir): {root_dir}")
    print(f"保存路径 (save_dir): {save_dir}\n")

    # --- 调用核心处理函数 ---
    # 这就是你指南中提到的函数，我们已经为它适配了本地路径
    convert_nc2npz(
        root_dir=root_dir,
        save_dir=save_dir,
        variables=["temperature", "geopotential"],  # 指定要处理的变量
        start_train_year=1979,
        start_val_year=2015,
        start_test_year=2017,
        end_year=2018,
        num_shards=16  # 将数据切成16个分片
    )

    print("\n=======================================")
    print("数据处理阶段成功结束！")
    print(f"请检查 '{save_dir}' 文件夹，确认 .npz 文件已生成。")


if __name__ == "__main__":
    main()
