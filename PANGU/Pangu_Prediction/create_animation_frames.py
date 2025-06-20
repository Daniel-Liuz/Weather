import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. Master Configuration ---

# <<<--- 总开关 1: 选择要绘制的图像类型 ---<<<
PLOT_TYPE = 'temperature'  # 可选值: 'weather_system' 或 'temperature'

# <<<--- 总开关 2: 选择要处理的预报模式 ---<<<
MODEL_TYPE_TO_PROCESS = '24h'  # 可选值: '1h', '3h', '6h', '24h'

# --- 自动路径配置 (无需修改) ---
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'DATA')
BASE_FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'FRAMES')

MODEL_PATHS = {
    '1h': {'input_dir': os.path.join(BASE_DATA_DIR, 'animation_data_1h')},
    '3h': {'input_dir': os.path.join(BASE_DATA_DIR, 'animation_data_3h')},
    '6h': {'input_dir': os.path.join(BASE_DATA_DIR, 'animation_data_6h')},
    '24h': {'input_dir': os.path.join(BASE_DATA_DIR, 'animation_data_24h')}
}
# 根据选择的绘图类型，自动确定输出子目录
config = MODEL_PATHS[MODEL_TYPE_TO_PROCESS]
config['output_dir'] = os.path.join(BASE_FRAMES_DIR, f"frames_{MODEL_TYPE_TO_PROCESS}_{PLOT_TYPE}")


# --- 2. Core Plotting Functions ---

def create_base_map():
    """创建一个带有标准地理背景的地图。"""
    fig = plt.figure(figsize=(15, 7.5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='#c9c9c9')
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='#a9a9a9')
    ax.coastlines(zorder=1)
    ax.gridlines(draw_labels=False, linestyle='--', color='gray')
    return fig, ax


def plot_weather_system_map(ax, lons, lats, surface_data):
    """在给定的地图上绘制等压线和风场。"""
    mslp = surface_data[0, :, :] / 100  # Pa -> hPa
    u10 = surface_data[1, :, :]
    v10 = surface_data[2, :, :]

    # 绘制海平面气压等值线
    contour_levels = np.arange(950, 1051, 4)
    contours = ax.contour(lons, lats, mslp, levels=contour_levels, colors='black', linewidths=1.0,
                          transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=8, fmt='%d')

    # 绘制风场 (降采样)
    step = 30
    ax.barbs(lons[::step], lats[::step], u10[::step, ::step], v10[::step, ::step],
             length=6, transform=ccrs.PlateCarree(), zorder=2)
    return "MSLP (hPa) and 10m Wind"


def plot_temperature_map(ax, lons, lats, surface_data):
    """在给定的地图上用彩色填充绘制温度。"""
    t2m_k = surface_data[3, :, :]  # T2M in Kelvin
    t2m_c = t2m_k - 273.15  # Convert to Celsius

    # 用 xarray 封装，便于绘图
    t2m_da = xr.DataArray(t2m_c, coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon'])

    # 绘制彩色填充图
    t2m_da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        vmin=-40, vmax=40,  # 固定色标范围，便于动画中对比
        cbar_kwargs={'label': '2-meter Temperature (°C)', 'shrink': 0.7, 'orientation': 'horizontal', 'pad': 0.05}
    )
    return "2m Temperature (°C)"


# --- 3. Main Execution Block ---

def main():
    input_dir = config['input_dir']
    output_dir = config['output_dir']

    print(f"--- Weather Map Frame Generation ---")
    print(f"模式: {MODEL_TYPE_TO_PROCESS}")
    print(f"绘图类型: {PLOT_TYPE}")
    print(f"读取NPY数据从: {input_dir}")
    print(f"保存PNG图像到: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(input_dir, 'mslp-*.npy')
    npy_files = sorted(glob.glob(search_pattern))

    if not npy_files:
        print(f"警告: 在目录 {input_dir} 中没有找到任何匹配 'mslp-*.npy' 的文件。")
        return

    print(f"找到 {len(npy_files)} 个数据帧，开始处理...")

    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 360, 1440, endpoint=False)

    for i, npy_path in enumerate(npy_files):
        frame_num = i + 1
        print(f"  处理帧 {frame_num}/{len(npy_files)}: {os.path.basename(npy_path)}")

        surface_data = np.load(npy_path)

        fig, ax = create_base_map()

        # 根据 PLOT_TYPE 调用不同的绘图函数
        if PLOT_TYPE == 'weather_system':
            plot_title_variable = plot_weather_system_map(ax, lons, lats, surface_data)
        elif PLOT_TYPE == 'temperature':
            plot_title_variable = plot_temperature_map(ax, lons, lats, surface_data)
        else:
            print(f"错误: 未知的 PLOT_TYPE '{PLOT_TYPE}'。")
            return

        # 添加统一的标题
        filename = os.path.basename(npy_path)
        timestamp = filename.split('-')[-1].replace('.npy', '')
        title_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:00 UTC"
        plt.title(f"Pangu-Weather Forecast: {plot_title_variable}\n{title_time}", fontsize=16, pad=20)

        # 保存图像
        output_image_path = os.path.join(output_dir, f'frame_{frame_num:03d}.png')
        plt.savefig(output_image_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"\n--- 所有 {PLOT_TYPE} 图像帧已成功生成！ ---")
    print(f"请在以下目录检查您的 .png 文件: {output_dir}")


if __name__ == "__main__":
    main()
