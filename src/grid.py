"""
空间网格划分模块
将地理空间划分为均匀网格，统计每个网格内的轨迹点密度，
并计算网格敏感度分数，用于后续自适应隐私预算分配。
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict, List


@dataclass
class GridCell:
    """网格单元"""
    row: int
    col: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    point_count: int = 0
    density: float = 0.0          # 归一化密度 [0, 1]
    sensitivity: float = 0.0      # 敏感度分数 [0, 1]，密度越高越敏感
    privacy_budget: float = 0.0   # 分配的隐私预算 epsilon_i


@dataclass
class SpatialGrid:
    """空间网格结构"""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_rows: int
    n_cols: int
    cells: Dict[Tuple[int, int], GridCell] = field(default_factory=dict)

    @property
    def lat_step(self) -> float:
        return (self.lat_max - self.lat_min) / self.n_rows

    @property
    def lon_step(self) -> float:
        return (self.lon_max - self.lon_min) / self.n_cols

    def get_cell_index(self, lat: float, lon: float) -> Tuple[int, int]:
        """根据经纬度获取网格索引 (row, col)"""
        row = int((lat - self.lat_min) / self.lat_step)
        col = int((lon - self.lon_min) / self.lon_step)
        row = max(0, min(row, self.n_rows - 1))
        col = max(0, min(col, self.n_cols - 1))
        return row, col

    def get_cell(self, lat: float, lon: float) -> GridCell:
        """获取坐标所在网格单元"""
        idx = self.get_cell_index(lat, lon)
        return self.cells.get(idx)


def build_spatial_grid(
    df: pd.DataFrame,
    n_rows: int = 20,
    n_cols: int = 20,
    lat_min: float = None,
    lat_max: float = None,
    lon_min: float = None,
    lon_max: float = None,
) -> SpatialGrid:
    """
    构建空间网格并统计各网格内的轨迹点数量

    参数:
        df: 包含 lat, lon 列的轨迹 DataFrame
        n_rows: 纬度方向网格数
        n_cols: 经度方向网格数
        lat/lon 边界: 若为 None 则从数据自动推断（略加 padding）

    返回:
        SpatialGrid 对象，已填充 point_count 和 density
    """
    # 边界推断
    if lat_min is None: lat_min = df["lat"].min() - 0.005
    if lat_max is None: lat_max = df["lat"].max() + 0.005
    if lon_min is None: lon_min = df["lon"].min() - 0.005
    if lon_max is None: lon_max = df["lon"].max() + 0.005

    grid = SpatialGrid(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        n_rows=n_rows, n_cols=n_cols,
    )

    # 初始化所有网格单元
    for r in range(n_rows):
        for c in range(n_cols):
            cell_lat_min = lat_min + r * grid.lat_step
            cell_lat_max = cell_lat_min + grid.lat_step
            cell_lon_min = lon_min + c * grid.lon_step
            cell_lon_max = cell_lon_min + grid.lon_step
            grid.cells[(r, c)] = GridCell(
                row=r, col=c,
                lat_min=cell_lat_min, lat_max=cell_lat_max,
                lon_min=cell_lon_min, lon_max=cell_lon_max,
            )

    # 统计每个网格的轨迹点数量
    lats = df["lat"].values
    lons = df["lon"].values

    for lat, lon in zip(lats, lons):
        idx = grid.get_cell_index(lat, lon)
        if idx in grid.cells:
            grid.cells[idx].point_count += 1

    # 计算归一化密度
    counts = np.array([cell.point_count for cell in grid.cells.values()])
    max_count = counts.max() if counts.max() > 0 else 1
    total_count = counts.sum() if counts.sum() > 0 else 1

    for cell in grid.cells.values():
        cell.density = cell.point_count / max_count  # [0,1] 归一化

    print(f"[网格] 构建 {n_rows}×{n_cols} 网格完成")
    print(f"[网格] 总点数: {int(total_count)}, 最大单格点数: {int(max_count)}")
    print(f"[网格] 非空格数: {int((counts > 0).sum())} / {n_rows * n_cols}")

    return grid


def compute_sensitivity_scores(
    grid: SpatialGrid,
    method: str = "density",
    smoothing: float = 0.1,
) -> SpatialGrid:
    """
    计算每个网格单元的敏感度分数

    参数:
        grid: 已统计点数的 SpatialGrid
        method: 
            'density'    -- 直接用密度作为敏感度（密度高 = 更敏感）
            'entropy'    -- 基于局部熵（考虑邻域）
            'percentile' -- 基于百分位分级
        smoothing: 拉普拉斯平滑系数，避免零密度

    返回:
        更新了 sensitivity 字段的 SpatialGrid
    """
    densities = np.array([
        grid.cells[(r, c)].density
        for r in range(grid.n_rows)
        for c in range(grid.n_cols)
    ]).reshape(grid.n_rows, grid.n_cols)

    if method == "density":
        # 密度归一化 + 平滑
        sensitivity_matrix = densities + smoothing
        sensitivity_matrix = (sensitivity_matrix - sensitivity_matrix.min()) / \
                             (sensitivity_matrix.max() - sensitivity_matrix.min() + 1e-10)

    elif method == "entropy":
        # 局部3×3邻域熵
        from scipy.ndimage import uniform_filter
        smoothed = uniform_filter(densities, size=3, mode='reflect') + smoothing
        smoothed /= smoothed.sum()
        entropy_matrix = -smoothed * np.log(smoothed + 1e-12)
        sensitivity_matrix = (entropy_matrix - entropy_matrix.min()) / \
                             (entropy_matrix.max() - entropy_matrix.min() + 1e-10)

    elif method == "percentile":
        # 分位数分级：低密度=0.2，中密度=0.5，高密度=1.0
        flat = densities.flatten()
        p33 = np.percentile(flat[flat > 0], 33) if (flat > 0).any() else 0.3
        p66 = np.percentile(flat[flat > 0], 66) if (flat > 0).any() else 0.6
        sensitivity_matrix = np.where(
            densities >= p66, 1.0,
            np.where(densities >= p33, 0.5, 0.2)
        )
    else:
        raise ValueError(f"未知方法: {method}")

    # 写回网格单元
    for r in range(grid.n_rows):
        for c in range(grid.n_cols):
            grid.cells[(r, c)].sensitivity = float(sensitivity_matrix[r, c])

    return grid


def get_density_matrix(grid: SpatialGrid) -> np.ndarray:
    """返回密度矩阵 shape=(n_rows, n_cols)"""
    mat = np.zeros((grid.n_rows, grid.n_cols))
    for (r, c), cell in grid.cells.items():
        mat[r, c] = cell.density
    return mat


def get_sensitivity_matrix(grid: SpatialGrid) -> np.ndarray:
    """返回敏感度矩阵 shape=(n_rows, n_cols)"""
    mat = np.zeros((grid.n_rows, grid.n_cols))
    for (r, c), cell in grid.cells.items():
        mat[r, c] = cell.sensitivity
    return mat


def get_budget_matrix(grid: SpatialGrid) -> np.ndarray:
    """返回隐私预算矩阵 shape=(n_rows, n_cols)"""
    mat = np.zeros((grid.n_rows, grid.n_cols))
    for (r, c), cell in grid.cells.items():
        mat[r, c] = cell.privacy_budget
    return mat


if __name__ == "__main__":
    from data_loader import generate_synthetic_geolife_data
    df = generate_synthetic_geolife_data(n_trajectories=30)
    grid = build_spatial_grid(df, n_rows=20, n_cols=20)
    grid = compute_sensitivity_scores(grid, method="density")
    sens_mat = get_sensitivity_matrix(grid)
    print(f"敏感度矩阵统计: min={sens_mat.min():.3f}, max={sens_mat.max():.3f}, mean={sens_mat.mean():.3f}")
