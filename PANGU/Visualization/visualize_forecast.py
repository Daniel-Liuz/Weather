import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

# --- 1. 用户配置区 ---
forecast_surface_npy_path = r"E:\Project_Collection\2025\WEATHER\PANGU\DATA\6h\npy\model_output_6h\mslp-10mU-10mV-2mT-20250605T00.npy"

# 从文件名中提取时间戳，用于图表标题
file_name = os.path.basename(forecast_surface_npy_path)
timestamp = file_name.split('-')[-1].replace('.npy', '') # 结果是 "20250605T00"


# --- 2. 数据加载与处理 ---

print(f"正在加载数据: {file_name}")

# 加载 .npy 文件
# 这是一个形状为 (4, 721, 1440) 的NumPy数组
surface_data = np.load(forecast_surface_npy_path)
print(f"数据加载成功，形状为: {surface_data.shape}")

# Pangu地表变量顺序:
# 索引 0: Mean sea level pressure (MSLP)
# 索引 1: 10-meter U-wind component (U10)
# 索引 2: 10-meter V-wind component (V10)
# 索引 3: 2-meter Temperature (T2M) in Kelvin

# 我们要绘制的是2米温度，所以我们提取索引为 3 的数据
t2m_grid = surface_data[3, :, :]
print(f"已提取2米温度数据，形状为: {t2m_grid.shape}")

# --- 3. 为数据添加地理坐标 ---
# 创建纬度和经度坐标轴
# 纬度从 90 (北极) 到 -90 (南极)，共721个点
lats = np.linspace(90, -90, 721)
# 经度从 0 到 359.75 (全球范围)，共1440个点
lons = np.linspace(0, 360, 1440, endpoint=False) # 使用endpoint=False更精确

# 使用 xarray 将纯粹的NumPy数组封装成带有坐标的 DataArray
# 这是进行地理信息可视化的关键一步
t2m_da = xr.DataArray(
    t2m_grid,
    coords={'lat': lats, 'lon': lons},
    dims=['lat', 'lon'],
    name='t2m'  # 变量名
)

# 盘古模型的温度单位是开尔文(K)，我们将其转换为摄氏度(°C)以便于理解
t2m_da_celsius = t2m_da - 273.15
t2m_da_celsius.attrs['units'] = '°C' # 给数据添加单位属性

print("已将数据封装为带坐标的 xarray.DataArray 并转换为摄氏度:")
print(t2m_da_celsius)


# --- 4. 使用 Matplotlib 和 Cartopy 绘制地图 ---

print("\n正在绘制全球温度地图...")

# 创建一个图形和一个带有地图投影的坐标轴
# figsize可以控制图像的大小
fig = plt.figure(figsize=(15, 7.5))
# ccrs.Robinson() 是一种常用的、看起来比较美观的全球投影
ax = plt.axes(projection=ccrs.Robinson())

# 设置全球范围
ax.set_global()

# 绘制数据。pcolormesh 是绘制网格数据的常用方法。
# vmin 和 vmax 可以手动设置色标的范围，以突出显示特定区域的温度变化
plot = t2m_da_celsius.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(), # 告诉Cartopy我们的数据是标准的经纬度格式
    cmap='coolwarm', # 使用 'coolwarm' 色彩方案，蓝色冷，红色暖
    cbar_kwargs={'label': '2-meter Temperature (°C)', 'shrink': 0.7} # 配置颜色条
)

# 添加地理元素，让地图更完整
ax.coastlines() # 绘制海岸线
ax.gridlines(draw_labels=False) # 添加经纬网格线，但不显示标签
ax.add_feature(cartopy.feature.BORDERS, linestyle=':') # 添加国界线

# 添加标题
title_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:00 UTC"
plt.title(f"Pangu-Weather Forecast: Global 2-meter Temperature\n{title_time}", fontsize=16)

# 显示图像
print("绘图完成！显示图像...")
plt.show()
