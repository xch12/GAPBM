"""
评估指标模块
============
计算以下指标，用于对比 GAPBM 与传统 Laplace DP 的性能：

1. 空间误差指标（数据可用性）
   - MAE  : 平均绝对误差（度）
   - RMSE : 均方根误差（度）
   - 平均 Haversine 距离（米）

2. 轨迹形态保持性指标
   - 轨迹长度比 (TLR)       : 扰动轨迹总长度 / 原始轨迹总长度
   - 方向偏差 (DDA)         : 平均相邻段方向偏差角
   - 平均位移均方根 (MSDR)  : 形态相似度

3. 热力图分布相似度
   - KL 散度 (KL Divergence)
   - 联合直方图 Pearson 相关系数
"""

import numpy as np
from typing import List, Tuple
from math import radians, cos, sin, asin, sqrt


# ─────────────────────────────────────────────
# 距离工具
# ─────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间 Haversine 距离（米）"""
    R = 6371000.0  # 地球半径（米）
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 2 * R * asin(sqrt(a))


def trajectory_length_m(traj: np.ndarray) -> float:
    """计算轨迹总长度（米），traj shape=(N,2) [lat,lon]"""
    total = 0.0
    for i in range(len(traj) - 1):
        total += haversine_m(traj[i, 0], traj[i, 1], traj[i+1, 0], traj[i+1, 1])
    return total


# ─────────────────────────────────────────────
# 1. 空间误差指标
# ─────────────────────────────────────────────

def compute_mae(original: List[np.ndarray], perturbed: List[np.ndarray]) -> float:
    """逐点平均绝对误差（度）"""
    errors = []
    for orig, pert in zip(original, perturbed):
        n = min(len(orig), len(pert))
        diff = np.abs(orig[:n] - pert[:n])  # shape=(n,2)
        errors.extend(diff.flatten().tolist())
    return float(np.mean(errors))


def compute_rmse(original: List[np.ndarray], perturbed: List[np.ndarray]) -> float:
    """逐点均方根误差（度）"""
    sq_errors = []
    for orig, pert in zip(original, perturbed):
        n = min(len(orig), len(pert))
        diff = (orig[:n] - pert[:n]) ** 2
        sq_errors.extend(diff.flatten().tolist())
    return float(np.sqrt(np.mean(sq_errors)))


def compute_mean_haversine(original: List[np.ndarray], perturbed: List[np.ndarray]) -> float:
    """逐点平均 Haversine 距离（米）"""
    distances = []
    for orig, pert in zip(original, perturbed):
        n = min(len(orig), len(pert))
        for i in range(n):
            d = haversine_m(orig[i, 0], orig[i, 1], pert[i, 0], pert[i, 1])
            distances.append(d)
    return float(np.mean(distances))


def compute_max_haversine(original: List[np.ndarray], perturbed: List[np.ndarray]) -> float:
    """逐点最大 Haversine 距离（米）"""
    distances = []
    for orig, pert in zip(original, perturbed):
        n = min(len(orig), len(pert))
        for i in range(n):
            d = haversine_m(orig[i, 0], orig[i, 1], pert[i, 0], pert[i, 1])
            distances.append(d)
    return float(np.max(distances)) if distances else 0.0


# ─────────────────────────────────────────────
# 2. 轨迹形态保持性指标
# ─────────────────────────────────────────────

def compute_trajectory_length_ratio(
    original: List[np.ndarray],
    perturbed: List[np.ndarray],
) -> float:
    """
    轨迹长度比 (TLR)
    TLR = mean(|L_perturbed - L_original| / L_original)
    越接近 0 表示长度保持越好
    """
    ratios = []
    for orig, pert in zip(original, perturbed):
        lo = trajectory_length_m(orig)
        lp = trajectory_length_m(pert)
        if lo > 0:
            ratios.append(abs(lp - lo) / lo)
    return float(np.mean(ratios)) if ratios else 0.0


def compute_direction_deviation(
    original: List[np.ndarray],
    perturbed: List[np.ndarray],
) -> float:
    """
    方向偏差角 (DDA) —— 度
    计算每对相邻点构成的向量之间的角度差均值
    越小表示轨迹方向保持越好
    """
    deviations = []
    for orig, pert in zip(original, perturbed):
        n = min(len(orig), len(pert)) - 1
        if n < 1:
            continue
        for i in range(n):
            v_o = orig[i+1] - orig[i]
            v_p = pert[i+1] - pert[i]
            norm_o = np.linalg.norm(v_o)
            norm_p = np.linalg.norm(v_p)
            if norm_o < 1e-12 or norm_p < 1e-12:
                continue
            cos_theta = np.clip(np.dot(v_o, v_p) / (norm_o * norm_p), -1, 1)
            angle = np.degrees(np.arccos(cos_theta))
            deviations.append(angle)
    return float(np.mean(deviations)) if deviations else 0.0


def compute_msdr(original: List[np.ndarray], perturbed: List[np.ndarray]) -> float:
    """
    平均位移均方根 (Mean Square Displacement Ratio)
    反映逐点形态保持程度（归一化到度）
    """
    sq_displacements = []
    for orig, pert in zip(original, perturbed):
        n = min(len(orig), len(pert))
        sq_displacements.extend(np.sum((orig[:n] - pert[:n]) ** 2, axis=1).tolist())
    return float(np.sqrt(np.mean(sq_displacements))) if sq_displacements else 0.0


# ─────────────────────────────────────────────
# 3. 热力图分布相似度
# ─────────────────────────────────────────────

def points_to_heatmap(
    trajectories: List[np.ndarray],
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    n_bins: int = 50,
) -> np.ndarray:
    """将轨迹点集汇总为二维密度直方图（热力图矩阵）"""
    all_lats = np.concatenate([t[:, 0] for t in trajectories])
    all_lons = np.concatenate([t[:, 1] for t in trajectories])

    heatmap, _, _ = np.histogram2d(
        all_lats, all_lons,
        bins=n_bins,
        range=[[lat_min, lat_max], [lon_min, lon_max]],
    )
    return heatmap


def compute_kl_divergence(
    original: List[np.ndarray],
    perturbed: List[np.ndarray],
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    n_bins: int = 50,
) -> float:
    """KL散度 D_KL(P_orig || P_pert)，越小表示分布越相似"""
    hm_orig = points_to_heatmap(original, lat_min, lat_max, lon_min, lon_max, n_bins)
    hm_pert = points_to_heatmap(perturbed, lat_min, lat_max, lon_min, lon_max, n_bins)

    # 归一化为概率分布
    eps = 1e-10
    p = (hm_orig + eps) / (hm_orig + eps).sum()
    q = (hm_pert + eps) / (hm_pert + eps).sum()

    kl = float(np.sum(p * np.log(p / q)))
    return kl


def compute_heatmap_correlation(
    original: List[np.ndarray],
    perturbed: List[np.ndarray],
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    n_bins: int = 50,
) -> float:
    """热力图 Pearson 相关系数（越接近1越好）"""
    hm_orig = points_to_heatmap(original, lat_min, lat_max, lon_min, lon_max, n_bins).flatten()
    hm_pert = points_to_heatmap(perturbed, lat_min, lat_max, lon_min, lon_max, n_bins).flatten()

    if hm_orig.std() < 1e-10 or hm_pert.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(hm_orig, hm_pert)[0, 1])


# ─────────────────────────────────────────────
# 汇总评估函数
# ─────────────────────────────────────────────

def compute_sensitive_zone_noise(
    original: List[np.ndarray],
    perturbed: List[np.ndarray],
    grid,
    sensitivity_threshold: float = 0.6,
) -> Tuple[float, float]:
    """
    计算高敏感区域 vs 低敏感区域的平均扰动幅度（度）
    返回 (sensitive_error, nonsensitive_error)
    GAPBM 预期：sensitive_error 更大（保护更强），nonsensitive_error 更小（扰动更少）
    """
    sens_errors, nonsens_errors = [], []
    for orig, pert in zip(original, perturbed):
        n = min(len(orig), len(pert))
        for i in range(n):
            cell = grid.get_cell(orig[i, 0], orig[i, 1])
            err = np.sqrt(np.sum((orig[i] - pert[i]) ** 2))
            if cell is not None and cell.sensitivity >= sensitivity_threshold:
                sens_errors.append(err)
            else:
                nonsens_errors.append(err)
    se = float(np.mean(sens_errors)) if sens_errors else 0.0
    nse = float(np.mean(nonsens_errors)) if nonsens_errors else 0.0
    return se, nse


def compute_privacy_utility_balance(results_gapbm: dict, results_fixed: dict) -> dict:
    """
    计算隐私-可用性平衡分数：
    - 误差越小 → 可用性越高
    - 热力图相关越高 → 分布保持越好
    综合评分 = Heatmap_Corr / (MAE + epsilon)
    """
    def score(r):
        mae = r["MAE(°)"] + 1e-10
        corr = max(r["Heatmap_Corr"], 0)
        return corr / mae
    return {
        "GAPBM_balance_score": round(score(results_gapbm), 4),
        "FixedDP_balance_score": round(score(results_fixed), 4),
    }


def evaluate_all(
    original: List[np.ndarray],
    perturbed_gapbm: List[np.ndarray],
    perturbed_fixed: List[np.ndarray],
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    grid=None,
) -> dict:
    """
    对 GAPBM 和传统 Laplace DP 进行全面评估
    返回包含所有指标的字典
    """
    results = {}
    for name, pert in [("GAPBM", perturbed_gapbm), ("FixedDP", perturbed_fixed)]:
        r = {}
        r["MAE(°)"]         = round(compute_mae(original, pert), 6)
        r["RMSE(°)"]        = round(compute_rmse(original, pert), 6)
        r["MeanDist(m)"]    = round(compute_mean_haversine(original, pert), 2)
        r["MaxDist(m)"]     = round(compute_max_haversine(original, pert), 2)
        r["TLR"]            = round(compute_trajectory_length_ratio(original, pert), 4)
        r["DDA(°)"]         = round(compute_direction_deviation(original, pert), 4)
        r["MSDR(°)"]        = round(compute_msdr(original, pert), 6)
        r["KL_Div"]         = round(compute_kl_divergence(
                                original, pert, lat_min, lat_max, lon_min, lon_max), 6)
        r["Heatmap_Corr"]   = round(compute_heatmap_correlation(
                                original, pert, lat_min, lat_max, lon_min, lon_max), 6)
        # 敏感/非敏感区域分区误差
        if grid is not None:
            se, nse = compute_sensitive_zone_noise(original, pert, grid)
            r["SensZone_Err"]     = round(se, 6)   # 高敏感区扰动（越大=隐私保护越强）
            r["NonSensZone_Err"]  = round(nse, 6)  # 低敏感区扰动（越小=可用性越好）
        results[name] = r

    return results


def print_evaluation_table(results: dict):
    """打印评估结果对比表格"""
    metrics = list(next(iter(results.values())).keys())
    methods = list(results.keys())

    col_w = 14
    header = f"{'指标':<22}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * (22 + col_w * len(methods)))
    print("  GAPBM vs 传统Laplace DP 评估对比")
    print("=" * (22 + col_w * len(methods)))
    print(header)
    print("-" * (22 + col_w * len(methods)))

    # 哪些指标越大越好
    bigger_is_better_set = {"Heatmap_Corr", "SensZone_Err"}
    # SensZone_Err: GAPBM应更大（对敏感区保护更强=更大扰动）

    better_count = {m: 0 for m in methods}
    for metric in metrics:
        row = f"{metric:<22}"
        vals = {m: results[m][metric] for m in methods}
        bigger_is_better = metric in bigger_is_better_set
        if bigger_is_better:
            best = max(methods, key=lambda m: vals[m])
        else:
            best = min(methods, key=lambda m: vals[m])
        better_count[best] = better_count.get(best, 0) + 1

        for m in methods:
            marker = " ✓" if m == best else "  "
            row += f"{vals[m]:>{col_w-2}.4f}{marker}"
        print(row)

    print("-" * (22 + col_w * len(methods)))
    print(f"{'综合优胜':<22}" + "".join(
        f"{better_count.get(m, 0):>{col_w}d}" for m in methods))
    print("=" * (22 + col_w * len(methods)))
    print("(✓ 表示该指标表现更优; SensZone_Err更大=对敏感区保护更强)\n")


if __name__ == "__main__":
    # 简单单元测试
    traj1 = np.array([[39.90 + i * 0.001, 116.40 + i * 0.001] for i in range(20)])
    traj2 = traj1 + np.random.normal(0, 0.002, traj1.shape)  # 轻微扰动
    traj3 = traj1 + np.random.normal(0, 0.005, traj1.shape)  # 较大扰动

    results = evaluate_all(
        [traj1], [traj2], [traj3],
        39.75, 40.10, 116.20, 116.60
    )
    print_evaluation_table(results)
