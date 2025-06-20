import xarray as xr
import numpy as np
import os

# --- 用户定义的预处理路径 ---
# 输入的NetCDF文件
surface_nc_file = r"E:\Project_Collection\2025\WEATHER\PANGU\DATA\6h\nc\era5_surface_2025_06.nc"
upper_nc_file = r"E:\Project_Collection\2025\WEATHER\PANGU\DATA\6h\nc\era5_upper_2025_06.nc"  # 确保此路径中没有隐藏字符

# .npy文件的输出目录
# 如果您希望存储在不同位置，可以更改这些路径
base_project_dir = r"E:\Project_Collection\2025\WEATHER\PANGU\DATA\6h\npy"   # 建议的基础项目目录
save_dir_surface_npy = os.path.join(base_project_dir, "processed_npy_data", "surface_data_6h")
save_dir_upper_npy = os.path.join(base_project_dir, "processed_npy_data", "upper_data_6h")

# 如果目录不存在则创建目录
os.makedirs(save_dir_surface_npy, exist_ok=True)
os.makedirs(save_dir_upper_npy, exist_ok=True)

# --- 第1部分：处理地表数据为.npy格式 ---
print(f"Processing surface data from: {surface_nc_file}")
df_surface = xr.open_dataset(surface_nc_file)
time_list_surface = np.array(df_surface.valid_time)

# Pangu-Weather期望的变量顺序: (MSLP, U10, V10, T2M)
# 请确保您的.nc文件使用诸如 'msl', 'u10', 'v10', 't2m' 之类的变量名
# 或者相应地调整下面的 df_surface.<var_name> 调用。
for i in range(len(time_list_surface)):
    time = time_list_surface[i]
    # 确保变量名与您的NetCDF文件中的名称匹配
    # 注意：ERA5通常使用'msl'（平均海平面气压），'u10'（10米U风分量），'v10'（10米V风分量），'t2m'（2米温度）
    msl = np.expand_dims(df_surface.msl.sel(valid_time=time), 0)
    u10 = np.expand_dims(df_surface.u10.sel(valid_time=time), 0)
    v10 = np.expand_dims(df_surface.v10.sel(valid_time=time), 0)
    t2m = np.expand_dims(df_surface.t2m.sel(valid_time=time), 0)

    res_array = np.concatenate([msl, u10, v10, t2m], axis=0)
    # 将 numpy.datetime64 转换为类似 'YYYY-MM-DDTHH' 的字符串
    time_str = np.datetime_as_string(time, unit='h').replace(':', '').replace('-', '')

    strname = f'mslp-10mU-10mV-2mT-{time_str}.npy'
    save_path = os.path.join(save_dir_surface_npy, strname)
    np.save(save_path, res_array.astype(np.float32))  # Ensure float32
    print(f"{strname} saved successfully to {save_dir_surface_npy}")

print("\nSurface data processing complete.\n")

# --- Part 2: 处理upper-air的数据为.npy的格式
print(f"Processing upper-air data from: {upper_nc_file}")
df_upper = xr.open_dataset(upper_nc_file)
time_list_upper = np.array(df_upper.valid_time)  # Or 'time' if that's the coordinate name

# 期望的变量顺序: (Z, Q, T, U, V)
# 期望的气压层顺序: (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa)
# 请确保您的.nc文件数据结构与此匹配，或进行相应调整。
# print(df_upper) # 取消注释以检查您的气压层数据集结构
# 检查气压层变量名，ERA5通常使用 'level' 或 'plev'

for i in range(len(time_list_upper)):
    time = time_list_upper[i]
    # 确保变量名与您的NetCDF文件中的名称匹配 (例如 'z', 'q', 't', 'u', 'v')
    # 同样，如果顺序不正确，请确保它们以正确的气压层顺序被选择。
    # Pangu模型期望气压层从1000hPa到50hPa排序。
    # 如果您的 'level' 或 'plev' 坐标是反向的，您可能需要对其排序：
    # .sel({level_coord_name: sorted(df_upper[level_coord_name].values, reverse=True)})
    # 或者如果已经是正确的顺序，则不需要排序。请根据您的数据确认。
    # ERA5 变量名通常是: 'z' (位势), 'q' (比湿), 't' (温度), 'u' (U风分量), 'v' (V风分量)
    z = np.expand_dims(df_upper.z.sel(valid_time=time), 0)  # Adjust 'z' if variable name is different
    q = np.expand_dims(df_upper.q.sel(valid_time=time), 0)  # Adjust 'q'
    t = np.expand_dims(df_upper.t.sel(valid_time=time), 0)  # Adjust 't'
    u = np.expand_dims(df_upper.u.sel(valid_time=time), 0)  # Adjust 'u'
    v = np.expand_dims(df_upper.v.sel(valid_time=time), 0)  # Adjust 'v'

    res_array = np.concatenate([z, q, t, u, v], axis=0)
    time_str = np.datetime_as_string(time, unit='h').replace(':', '').replace('-', '')

    strname = f'z-q-t-u-v-{time_str}.npy'
    save_path = os.path.join(save_dir_upper_npy, strname)
    np.save(save_path, res_array.astype(np.float32))  # Ensure float32
    print(f"{strname} saved successfully to {save_dir_upper_npy}")

print("\nUpper-air data processing complete.\n")
print(f"IMPORTANT: Note the generated .npy filenames (e.g., mslp-10mU-10mV-2mT-{time_str}.npy).")
print("You will need to specify one of these timestamps for the model running phase.")

