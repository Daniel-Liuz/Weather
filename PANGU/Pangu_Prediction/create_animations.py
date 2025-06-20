import os
import glob
import imageio

# --- 1. Configuration ---
# (这个脚本是全自动的，通常无需修改)

# 基础路径
BASE_DIR = os.path.dirname(__file__)
FRAMES_DIR = os.path.join(BASE_DIR, 'FRAMES')
ANIMATIONS_DIR = os.path.join(BASE_DIR, 'pangu_weather')

# GIF 参数
# duration: 每帧显示的时间（单位：毫秒）。500ms = 0.5秒/帧 (即 2 FPS)
# 您可以调小这个值让动画更快，例如 200 (即 5 FPS)
FRAME_DURATION_MS = 500
# loop: 循环次数。0 表示无限循环。
LOOP_COUNT = 0


# --- 2. Main Execution Block ---
def main():
    print("--- GIF Animation Synthesis Script ---")

    if not os.path.isdir(FRAMES_DIR):
        print(f"错误: 找不到 FRAMES 目录: {FRAMES_DIR}")
        print("请先运行 create_animation_frames.py 生成图片帧。")
        return

    # 创建用于存放GIF的目录
    os.makedirs(ANIMATIONS_DIR, exist_ok=True)
    print(f"动画将保存到: {ANIMATIONS_DIR}")

    # 获取 FRAMES 目录下的所有子文件夹
    frame_folders = [f for f in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, f))]

    if not frame_folders:
        print("警告: 在 FRAMES 目录下没有找到任何图片帧文件夹。")
        return

    print(f"\n找到 {len(frame_folders)} 个动画序列待处理...")

    for folder_name in frame_folders:
        image_folder_path = os.path.join(FRAMES_DIR, folder_name)
        print(f"\n[+] 正在处理: {folder_name}")

        # 查找文件夹内所有的 .png 文件并按名称排序
        search_pattern = os.path.join(image_folder_path, 'frame_*.png')
        image_files = sorted(glob.glob(search_pattern))

        if not image_files:
            print("  -> 未找到任何 .png 文件，跳过。")
            continue

        print(f"  -> 找到 {len(image_files)} 帧图片，开始合成GIF...")

        # 读取所有图片到内存
        images = []
        for filename in image_files:
            images.append(imageio.imread(filename))

        # 构建输出GIF文件名
        output_gif_name = folder_name.replace('frames_', 'animation_') + '.gif'
        output_gif_path = os.path.join(ANIMATIONS_DIR, output_gif_name)

        # 使用 imageio 保存为 GIF
        imageio.mimsave(output_gif_path, images, duration=FRAME_DURATION_MS, loop=LOOP_COUNT)

        print(f"  -> 成功！动画已保存至: {output_gif_path}")

    print("\n--- 所有动画合成完毕！---")


if __name__ == "__main__":
    main()

