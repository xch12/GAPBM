"""
GeoLife 轨迹数据加载与预处理模块
支持从微软GeoLife数据集加载 .plt 文件，并生成模拟数据用于测试
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# GeoLife 北京核心区域边界
BEIJING_BOUNDS = {
    "lat_min": 39.75,
    "lat_max": 40.10,
    "lon_min": 116.20,
    "lon_max": 116.60,
}


def load_plt_file(filepath: str) -> pd.DataFrame:
    """
    读取单个 GeoLife .plt 文件
    文件格式: 前6行为元信息，之后每行: lat, lon, 0, alt, days, date, time
    """
    try:
        df = pd.read_csv(
            filepath,
            skiprows=6,
            header=None,
            names=["lat", "lon", "zero", "alt", "days", "date", "time"],
            usecols=["lat", "lon", "alt", "date", "time"],
        )
        df = df.dropna()
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        return df[["lat", "lon"]]
    except Exception:
        return pd.DataFrame(columns=["lat", "lon"])


def load_geolife_data(data_dir: str, max_users: int = 10, max_files_per_user: int = 5) -> pd.DataFrame:
    """
    批量加载 GeoLife 数据集
    data_dir: GeoLife Data/ 目录路径
    """
    data_path = Path(data_dir)
    all_points = []

    if not data_path.exists():
        print(f"[警告] 数据目录不存在: {data_dir}，将使用模拟数据")
        return generate_synthetic_geolife_data()

    user_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])[:max_users]
    if not user_dirs:
        print("[警告] 未找到用户目录，将使用模拟数据")
        return generate_synthetic_geolife_data()

    print(f"[信息] 正在加载 {len(user_dirs)} 个用户的轨迹数据...")
    for user_dir in tqdm(user_dirs, desc="加载用户数据"):
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.exists():
            continue
        plt_files = sorted(traj_dir.glob("*.plt"))[:max_files_per_user]
        for plt_file in plt_files:
            df = load_plt_file(str(plt_file))
            if not df.empty:
                all_points.append(df)

    if not all_points:
        print("[警告] 未加载到有效数据，将使用模拟数据")
        return generate_synthetic_geolife_data()

    result = pd.concat(all_points, ignore_index=True)
    # 过滤北京区域
    result = filter_beijing_region(result)
    print(f"[信息] 共加载 {len(result)} 个轨迹点（北京区域）")
    return result


def filter_beijing_region(df: pd.DataFrame) -> pd.DataFrame:
    """过滤保留北京核心区域的轨迹点"""
    b = BEIJING_BOUNDS
    mask = (
        (df["lat"] >= b["lat_min"]) & (df["lat"] <= b["lat_max"]) &
        (df["lon"] >= b["lon_min"]) & (df["lon"] <= b["lon_max"])
    )
    return df[mask].reset_index(drop=True)


def generate_synthetic_geolife_data(n_trajectories: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟的 GeoLife 风格北京轨迹数据
    包含热点区域（天安门、中关村、朝阳、西城）和随机路径
    """
    np.random.seed(seed)
    b = BEIJING_BOUNDS

    # 北京典型热点区域 (lat, lon, 权重, 聚集半径)
    hotspots = [
        (39.9042, 116.4074, 3.0, 0.015),   # 天安门广场
        (39.9804, 116.3065, 2.5, 0.012),   # 中关村
        (39.9219, 116.4431, 2.0, 0.010),   # 朝阳公园
        (39.9274, 116.3785, 1.8, 0.008),   # 西单
        (39.8674, 116.4789, 1.5, 0.010),   # 北京南站
        (40.0024, 116.4432, 1.2, 0.012),   # 奥林匹克公园
        (39.9289, 116.5880, 1.0, 0.008),   # 首都机场方向
        (39.8836, 116.3913, 1.5, 0.009),   # 北京西站
    ]

    all_points = []

    for traj_id in range(n_trajectories):
        # 随机选择起始热点
        weights = np.array([h[2] for h in hotspots])
        weights /= weights.sum()
        start_idx = np.random.choice(len(hotspots), p=weights)
        end_idx = np.random.choice(len(hotspots), p=weights)

        start_lat = hotspots[start_idx][0] + np.random.normal(0, hotspots[start_idx][3])
        start_lon = hotspots[start_idx][1] + np.random.normal(0, hotspots[start_idx][3])
        end_lat = hotspots[end_idx][0] + np.random.normal(0, hotspots[end_idx][3])
        end_lon = hotspots[end_idx][1] + np.random.normal(0, hotspots[end_idx][3])

        n_points = np.random.randint(30, 120)
        t = np.linspace(0, 1, n_points)

        # 平滑插值 + 随机游走噪声
        noise_lat = np.cumsum(np.random.normal(0, 0.0004, n_points))
        noise_lon = np.cumsum(np.random.normal(0, 0.0004, n_points))

        lats = start_lat + (end_lat - start_lat) * t + noise_lat * 0.3
        lons = start_lon + (end_lon - start_lon) * t + noise_lon * 0.3

        traj_df = pd.DataFrame({"lat": lats, "lon": lons, "traj_id": traj_id})
        all_points.append(traj_df)

    df = pd.concat(all_points, ignore_index=True)

    # 裁剪到北京范围
    df["lat"] = df["lat"].clip(b["lat_min"], b["lat_max"])
    df["lon"] = df["lon"].clip(b["lon_min"], b["lon_max"])

    print(f"[信息] 生成模拟轨迹数据: {n_trajectories} 条轨迹，共 {len(df)} 个轨迹点")
    return df


def get_trajectories_list(df: pd.DataFrame) -> list:
    """
    将 DataFrame 按 traj_id 分组，返回轨迹列表
    每个元素为 np.array shape=(N, 2) [lat, lon]
    """
    if "traj_id" not in df.columns:
        # 将所有点视为单条轨迹
        return [df[["lat", "lon"]].values]

    trajectories = []
    for tid, group in df.groupby("traj_id"):
        pts = group[["lat", "lon"]].values
        if len(pts) >= 5:
            trajectories.append(pts)
    return trajectories


if __name__ == "__main__":
    # 测试数据加载
    df = generate_synthetic_geolife_data(n_trajectories=20)
    print(df.head())
    print(f"数据形状: {df.shape}")
    print(f"纬度范围: [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    print(f"经度范围: [{df['lon'].min():.4f}, {df['lon'].max():.4f}]")
