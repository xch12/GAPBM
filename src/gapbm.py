"""
GAPBM 核心模块：自适应隐私预算分配 + Laplace 噪声扰动
=======================================================
GAPBM (Grid-based Adaptive Privacy Budget Mechanism) 算法流程：

1. 基于空间网格的密度统计得到敏感度矩阵 S(i,j)
2. 自适应预算分配：
      ε_total = 总预算
      敏感区域（高密度）→ 分配更小的 ε（更强保护）
      普通区域（低密度）→ 分配更大的 ε（更少扰动）
   
   具体公式：
      w(i,j) = 1 - S(i,j)           # 反敏感度权重
      ε(i,j) = ε_total * w(i,j) / Σw  * n_cells  （再归一化使均值=ε_total）

3. 对每个轨迹点 p=(lat, lon)，查找其所在网格，
   按该网格的 ε(i,j) 使用 Laplace 机制添加噪声：
      scale = Δf / ε(i,j)   其中 Δf = 全局敏感度（经纬度范围）
      p' = p + Laplace(0, scale)

传统 Laplace DP（基线）：
   对所有轨迹点使用固定 ε，scale = Δf / ε
"""

import numpy as np
from typing import List, Tuple, Optional
from grid import SpatialGrid, get_budget_matrix


# ─────────────────────────────────────────────
# 全局敏感度（地理坐标下的最大位移）
# 对应北京区域 0.35° 纬度 × 0.40° 经度
# ─────────────────────────────────────────────
GLOBAL_SENSITIVITY_LAT = 0.35   # 度
GLOBAL_SENSITIVITY_LON = 0.40   # 度


# ══════════════════════════════════════════════
# Part 1: 自适应预算分配
# ══════════════════════════════════════════════

def allocate_adaptive_budget(
    grid: SpatialGrid,
    epsilon_total: float = 1.0,
    min_epsilon_ratio: float = 0.1,
    max_epsilon_ratio: float = 3.0,
    allocation_strategy: str = "inverse_sensitivity",
) -> SpatialGrid:
    """
    自适应隐私预算分配

    参数：
        grid             : 已完成敏感度计算的 SpatialGrid
        epsilon_total    : 总隐私预算 ε
        min_epsilon_ratio: 单格最小预算比率（相对均值）
        max_epsilon_ratio: 单格最大预算比率（相对均值）
        allocation_strategy:
            'inverse_sensitivity' -- 反比于敏感度（主方法）
            'exponential'         -- 指数衰减
            'step'                -- 三档离散分配

    返回：
        更新了 privacy_budget 字段的 SpatialGrid
    """
    n = grid.n_rows * grid.n_cols
    mean_eps = epsilon_total  # 均值目标

    # 收集敏感度
    sens = np.array([
        grid.cells[(r, c)].sensitivity
        for r in range(grid.n_rows)
        for c in range(grid.n_cols)
    ])  # shape = (n,)

    if allocation_strategy == "inverse_sensitivity":
        # w_i = 1 - S_i  （高敏感 → 低权重 → 小 ε → 强保护）
        weights = 1.0 - sens
        weights = np.clip(weights, 0.05, 1.0)  # 避免零权重
        # 归一化使总预算 = n * epsilon_total
        weights = weights / weights.mean()
        epsilons = epsilon_total * weights

    elif allocation_strategy == "exponential":
        # ε_i = ε_total * exp(-α * S_i)
        alpha = 2.0
        epsilons = epsilon_total * np.exp(-alpha * sens)
        epsilons = epsilons / epsilons.mean() * epsilon_total

    elif allocation_strategy == "step":
        # 三档分配
        epsilons = np.where(
            sens >= 0.7, epsilon_total * 0.3,     # 高敏感：小预算（强保护）
            np.where(sens >= 0.3, epsilon_total * 1.0,  # 中敏感：标准
                     epsilon_total * 2.5)          # 低敏感：大预算（弱保护）
        )
    else:
        raise ValueError(f"未知分配策略: {allocation_strategy}")

    # 约束到合理范围
    eps_min = mean_eps * min_epsilon_ratio
    eps_max = mean_eps * max_epsilon_ratio
    epsilons = np.clip(epsilons, eps_min, eps_max)

    # 写回网格
    idx = 0
    for r in range(grid.n_rows):
        for c in range(grid.n_cols):
            grid.cells[(r, c)].privacy_budget = float(epsilons[idx])
            idx += 1

    eps_arr = get_budget_matrix(grid)
    print(f"[GAPBM预算] ε_total={epsilon_total:.2f}, "
          f"min={eps_arr.min():.4f}, max={eps_arr.max():.4f}, "
          f"mean={eps_arr.mean():.4f}, std={eps_arr.std():.4f}")

    return grid


# ══════════════════════════════════════════════
# Part 2: Laplace 噪声机制
# ══════════════════════════════════════════════

def laplace_noise(scale: float, size: int = 1, rng: np.random.Generator = None) -> np.ndarray:
    """生成 Laplace 噪声，scale = Δf / ε"""
    if rng is None:
        rng = np.random.default_rng()
    return rng.laplace(loc=0.0, scale=scale, size=size)


def perturb_point_laplace(
    lat: float,
    lon: float,
    epsilon: float,
    sensitivity_lat: float = GLOBAL_SENSITIVITY_LAT,
    sensitivity_lon: float = GLOBAL_SENSITIVITY_LON,
    rng: np.random.Generator = None,
    lat_bounds: Tuple[float, float] = (39.75, 40.10),
    lon_bounds: Tuple[float, float] = (116.20, 116.60),
) -> Tuple[float, float]:
    """
    对单个坐标点施加 Laplace 差分隐私扰动

    返回：扰动后坐标 (lat', lon')，裁剪到合法范围
    """
    if rng is None:
        rng = np.random.default_rng()

    scale_lat = sensitivity_lat / max(epsilon, 1e-9)
    scale_lon = sensitivity_lon / max(epsilon, 1e-9)

    noisy_lat = lat + float(laplace_noise(scale_lat, 1, rng))
    noisy_lon = lon + float(laplace_noise(scale_lon, 1, rng))

    # 裁剪到边界
    noisy_lat = np.clip(noisy_lat, lat_bounds[0], lat_bounds[1])
    noisy_lon = np.clip(noisy_lon, lon_bounds[0], lon_bounds[1])

    return float(noisy_lat), float(noisy_lon)


# ══════════════════════════════════════════════
# Part 3: GAPBM 轨迹扰动
# ══════════════════════════════════════════════

def perturb_trajectory_gapbm(
    trajectory: np.ndarray,
    grid: SpatialGrid,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    GAPBM 方法扰动单条轨迹

    参数：
        trajectory: shape=(N, 2) [lat, lon]
        grid      : 已完成自适应预算分配的 SpatialGrid

    返回：扰动后轨迹 shape=(N, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    perturbed = np.zeros_like(trajectory)
    lat_bounds = (grid.lat_min, grid.lat_max)
    lon_bounds = (grid.lon_min, grid.lon_max)

    for i, (lat, lon) in enumerate(trajectory):
        cell = grid.get_cell(lat, lon)
        if cell is not None and cell.privacy_budget > 0:
            epsilon = cell.privacy_budget
        else:
            # 未命中网格，使用默认 ε=1.0
            epsilon = 1.0

        p_lat, p_lon = perturb_point_laplace(
            lat, lon, epsilon,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            rng=rng,
        )
        perturbed[i] = [p_lat, p_lon]

    return perturbed


def perturb_trajectories_gapbm(
    trajectories: List[np.ndarray],
    grid: SpatialGrid,
    seed: int = 42,
) -> List[np.ndarray]:
    """批量 GAPBM 扰动"""
    rng = np.random.default_rng(seed)
    return [perturb_trajectory_gapbm(traj, grid, rng) for traj in trajectories]


# ══════════════════════════════════════════════
# Part 4: 传统 Laplace DP 基线方法
# ══════════════════════════════════════════════

def perturb_trajectory_fixed_dp(
    trajectory: np.ndarray,
    epsilon: float,
    grid: SpatialGrid,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    传统固定预算差分隐私扰动单条轨迹
    所有点使用相同的 epsilon
    """
    if rng is None:
        rng = np.random.default_rng()

    perturbed = np.zeros_like(trajectory)
    lat_bounds = (grid.lat_min, grid.lat_max)
    lon_bounds = (grid.lon_min, grid.lon_max)

    for i, (lat, lon) in enumerate(trajectory):
        p_lat, p_lon = perturb_point_laplace(
            lat, lon, epsilon,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            rng=rng,
        )
        perturbed[i] = [p_lat, p_lon]

    return perturbed


def perturb_trajectories_fixed_dp(
    trajectories: List[np.ndarray],
    epsilon: float,
    grid: SpatialGrid,
    seed: int = 42,
) -> List[np.ndarray]:
    """批量传统 Laplace DP 扰动"""
    rng = np.random.default_rng(seed)
    return [perturb_trajectory_fixed_dp(traj, epsilon, grid, rng) for traj in trajectories]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import generate_synthetic_geolife_data
    from grid import build_spatial_grid, compute_sensitivity_scores

    df = generate_synthetic_geolife_data(n_trajectories=10, seed=0)
    grid = build_spatial_grid(df, n_rows=20, n_cols=20)
    grid = compute_sensitivity_scores(grid, method="density")
    grid = allocate_adaptive_budget(grid, epsilon_total=1.0)

    # 测试单点扰动
    test_traj = np.array([[39.90, 116.40], [39.91, 116.41], [39.92, 116.42]])
    pert = perturb_trajectory_gapbm(test_traj, grid)
    print("原始轨迹:", test_traj)
    print("GAPBM扰动:", pert)
    print("误差:", np.abs(pert - test_traj).mean(axis=0))
