"""
可视化模块
==========
提供以下可视化图表：

1. 空间网格密度热力图（原始轨迹点分布）
2. 隐私预算分配热力图（GAPBM vs 固定）
3. 轨迹对比图（原始 / GAPBM扰动 / FixedDP扰动）
4. 热力图对比（三列：原始 / GAPBM / FixedDP）
5. 评估指标条形图对比
6. 误差随 ε 变化曲线
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple

# 统一中文字体支持
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CMAP_HEAT = "YlOrRd"
CMAP_BUDGET = "RdYlGn"   # 红=小预算(强保护), 绿=大预算(弱保护)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, name: str, dpi: int = 150):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 已保存: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 1. 网格密度热力图
# ──────────────────────────────────────────────────────────────────────────────

def plot_grid_density(grid, title: str = "Grid Density Heatmap", filename: str = "grid_density.png"):
    """绘制网格密度热力图"""
    from grid import get_density_matrix
    density = get_density_matrix(grid)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        density, origin="lower", aspect="auto",
        cmap=CMAP_HEAT, interpolation="nearest",
        extent=[grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max],
    )
    plt.colorbar(im, ax=ax, label="Normalized Density")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    return _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 自适应隐私预算分配热力图
# ──────────────────────────────────────────────────────────────────────────────

def plot_budget_heatmap(grid, epsilon_total: float = 1.0, filename: str = "budget_heatmap.png"):
    """绘制自适应隐私预算分配热力图，标注敏感区域"""
    from grid import get_budget_matrix, get_sensitivity_matrix

    budget = get_budget_matrix(grid)
    sens   = get_sensitivity_matrix(grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：敏感度图
    im1 = axes[0].imshow(
        sens, origin="lower", aspect="auto", cmap="Reds",
        interpolation="nearest",
        extent=[grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max],
    )
    plt.colorbar(im1, ax=axes[0], label="Sensitivity Score")
    axes[0].set_title("Sensitivity Score Map\n(Higher = More Sensitive)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # 右：隐私预算图
    vmin = budget.min()
    vmax = budget.max()
    im2 = axes[1].imshow(
        budget, origin="lower", aspect="auto", cmap=CMAP_BUDGET,
        interpolation="nearest", vmin=vmin, vmax=vmax,
        extent=[grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max],
    )
    cb = plt.colorbar(im2, ax=axes[1], label=f"Privacy Budget ε (total={epsilon_total})")
    axes[1].set_title("Adaptive Privacy Budget Allocation\n(Red=Strong Protection, Green=Weak)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    # 标注高低预算区域
    budget_flat = budget.flatten()
    low_thresh  = np.percentile(budget_flat, 20)
    high_thresh = np.percentile(budget_flat, 80)
    axes[1].contour(
        np.linspace(grid.lon_min, grid.lon_max, grid.n_cols),
        np.linspace(grid.lat_min, grid.lat_max, grid.n_rows),
        budget,
        levels=[low_thresh, high_thresh],
        colors=["darkred", "darkgreen"],
        linewidths=1.2,
        linestyles=["--", "--"],
    )

    fig.suptitle("GAPBM: Sensitivity & Adaptive Budget", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 轨迹对比图
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory_comparison(
    original: List[np.ndarray],
    perturbed_gapbm: List[np.ndarray],
    perturbed_fixed: List[np.ndarray],
    grid,
    n_show: int = 5,
    filename: str = "trajectory_comparison.png",
):
    """绘制轨迹三列对比图：原始 / GAPBM / FixedDP"""
    n_show = min(n_show, len(original))
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 3.5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    titles = ["Original Trajectory", "GAPBM Perturbed", "Fixed-ε DP Perturbed"]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    datasets = [original, perturbed_gapbm, perturbed_fixed]

    for row in range(n_show):
        for col in range(3):
            ax = axes[row, col]
            traj = datasets[col][row]
            ax.plot(traj[:, 1], traj[:, 0], "-o",
                    color=colors[col], alpha=0.8, markersize=2, linewidth=1.5)
            ax.plot(traj[0, 1], traj[0, 0], "^", color="green", markersize=8, label="Start")
            ax.plot(traj[-1, 1], traj[-1, 0], "s", color="red",   markersize=8, label="End")
            ax.set_xlim(grid.lon_min, grid.lon_max)
            ax.set_ylim(grid.lat_min, grid.lat_max)
            if row == 0:
                ax.set_title(titles[col], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Traj #{row+1}\nLatitude")
            ax.set_xlabel("Longitude") if row == n_show - 1 else None
            ax.tick_params(labelsize=7)

    plt.suptitle("Trajectory Comparison: Original vs GAPBM vs Fixed-ε DP",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 热力图三列对比
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmap_comparison(
    original: List[np.ndarray],
    perturbed_gapbm: List[np.ndarray],
    perturbed_fixed: List[np.ndarray],
    grid,
    n_bins: int = 50,
    filename: str = "heatmap_comparison.png",
):
    """绘制原始/GAPBM/FixedDP 热力图对比"""
    from metrics import points_to_heatmap

    def make_hm(trajs):
        return points_to_heatmap(trajs, grid.lat_min, grid.lat_max,
                                  grid.lon_min, grid.lon_max, n_bins)

    hm_orig  = make_hm(original)
    hm_gapbm = make_hm(perturbed_gapbm)
    hm_fixed = make_hm(perturbed_fixed)

    vmax = max(hm_orig.max(), hm_gapbm.max(), hm_fixed.max())
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    data_list = [hm_orig, hm_gapbm, hm_fixed]
    labels = ["Original", "GAPBM", "Fixed-ε DP"]
    cmaps  = [CMAP_HEAT, CMAP_HEAT, CMAP_HEAT]

    for ax, data, label, cmap in zip(axes, data_list, labels, cmaps):
        im = ax.imshow(data, origin="lower", aspect="auto",
                       cmap=cmap, interpolation="gaussian",
                       vmin=0, vmax=vmax, extent=extent)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Point Count")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("Heatmap Distribution Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 5. 评估指标条形图
# ──────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar(results: dict, filename: str = "metrics_comparison.png"):
    """绘制 GAPBM vs FixedDP 各指标条形图"""
    methods = list(results.keys())
    metrics = list(results[methods[0]].keys())

    # 分组：误差类 vs 保持性类
    error_metrics = ["MAE(°)", "RMSE(°)", "MeanDist(m)", "MaxDist(m)", "MSDR(°)"]
    shape_metrics  = ["TLR", "DDA(°)", "KL_Div"]
    corr_metrics   = ["Heatmap_Corr"]

    groups = [
        ("Spatial Error (lower is better)", error_metrics),
        ("Shape Preservation (lower is better)", shape_metrics),
        ("Heatmap Correlation (higher is better)", corr_metrics),
    ]

    colors = {"GAPBM": "#1976D2", "FixedDP": "#F57C00"}
    x_gap = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (group_title, group_metrics) in zip(axes, groups):
        valid = [m for m in group_metrics if m in metrics]
        if not valid:
            ax.set_visible(False)
            continue

        x = np.arange(len(valid))
        width = 0.35
        for i, method in enumerate(methods):
            vals = [results[method].get(m, 0) for m in valid]
            bars = ax.bar(x + i * width - width / 2, vals, width,
                          label=method, color=colors.get(method, "#888"),
                          alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(valid, rotation=20, ha="right", fontsize=9)
        ax.set_title(group_title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("GAPBM vs Fixed-ε DP: Comprehensive Metric Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 6. 误差随 ε 变化曲线
# ──────────────────────────────────────────────────────────────────────────────

def plot_epsilon_vs_error(
    epsilon_list: List[float],
    gapbm_errors: List[float],
    fixed_errors: List[float],
    metric_name: str = "MeanDist(m)",
    filename: str = "epsilon_vs_error.png",
):
    """绘制不同总预算 ε 下的误差曲线对比"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epsilon_list, gapbm_errors, "b-o", label="GAPBM", linewidth=2, markersize=6)
    ax.plot(epsilon_list, fixed_errors, "r-s", label="Fixed-ε DP", linewidth=2, markersize=6)

    ax.fill_between(epsilon_list, gapbm_errors, fixed_errors,
                    where=[g < f for g, f in zip(gapbm_errors, fixed_errors)],
                    alpha=0.15, color="blue", label="GAPBM advantage")

    ax.set_xlabel("Privacy Budget ε (total)", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"{metric_name} vs Privacy Budget ε\n(GAPBM vs Fixed-ε DP)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 6b. 敏感区域 vs 普通区域扰动对比图
# ──────────────────────────────────────────────────────────────────────────────

def plot_zone_comparison(results: dict, filename: str = "zone_comparison.png"):
    """
    绘制敏感区域/非敏感区域扰动对比雷达/条形图
    核心图：展示 GAPBM 在敏感区更强保护、普通区更少扰动
    """
    if "SensZone_Err" not in results.get("GAPBM", {}):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    methods = list(results.keys())
    bar_colors = {"GAPBM": "#1976D2", "FixedDP": "#F57C00"}

    # 左图：敏感区 vs 普通区扰动对比
    ax = axes[0]
    zone_labels = ["Sensitive Zone\n(High Density)", "Non-Sensitive Zone\n(Low Density)"]
    x = np.arange(len(zone_labels))
    width = 0.35
    for j, m in enumerate(methods):
        vals = [results[m].get("SensZone_Err", 0), results[m].get("NonSensZone_Err", 0)]
        bars = ax.bar(x + j * width - width / 2, vals, width,
                      label=m, color=bar_colors.get(m, "#888"), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=10)
    ax.set_ylabel("Average Perturbation Error (°)", fontsize=10)
    ax.set_title("Zone-based Perturbation Analysis\n"
                 "(Sensitive↑=Stronger Privacy, Non-Sensitive↓=Better Utility)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 右图：GAPBM 的改进率条形图
    ax2 = axes[1]
    key_metrics = [m for m in ["MAE(°)", "RMSE(°)", "TLR", "KL_Div", "Heatmap_Corr",
                                "SensZone_Err", "NonSensZone_Err"]
                   if m in results.get("GAPBM", {})]

    improvements = []
    labels_imp = []
    for m in key_metrics:
        g_val = results["GAPBM"][m]
        f_val = results["FixedDP"][m]
        if m in ["Heatmap_Corr", "SensZone_Err"]:
            # 越大越好（GAPBM在敏感区应更大）
            imp = (g_val - f_val) / (abs(f_val) + 1e-9) * 100
        else:
            # 越小越好
            imp = (f_val - g_val) / (abs(f_val) + 1e-9) * 100
        improvements.append(imp)
        labels_imp.append(m)

    colors_imp = ["#2196F3" if v >= 0 else "#FF5722" for v in improvements]
    bars = ax2.barh(labels_imp, improvements, color=colors_imp, alpha=0.85)
    for bar, val in zip(bars, improvements):
        ax2.text(val + (0.5 if val >= 0 else -0.5),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.1f}%", va="center", fontsize=9,
                 ha="left" if val >= 0 else "right")

    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("GAPBM Improvement vs Fixed-ε DP (%)", fontsize=10)
    ax2.set_title("GAPBM Improvement Rate\n(Blue=Better, Red=Worse)", fontsize=10, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("GAPBM Privacy-Utility Balance Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 7. 综合摘要图
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(
    original: List[np.ndarray],
    perturbed_gapbm: List[np.ndarray],
    perturbed_fixed: List[np.ndarray],
    grid,
    results: dict,
    filename: str = "summary_dashboard.png",
):
    """综合摘要仪表板（2×3 子图）"""
    from metrics import points_to_heatmap

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 行1列1: 原始热力图
    ax1 = fig.add_subplot(gs[0, 0])
    hm_orig = points_to_heatmap(original, grid.lat_min, grid.lat_max,
                                 grid.lon_min, grid.lon_max, 40)
    im1 = ax1.imshow(hm_orig, origin="lower", cmap=CMAP_HEAT, aspect="auto",
                     extent=[grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max])
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    ax1.set_title("Original Trajectory Heatmap", fontweight="bold")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")

    # ── 行1列2: GAPBM热力图
    ax2 = fig.add_subplot(gs[0, 1])
    hm_gapbm = points_to_heatmap(perturbed_gapbm, grid.lat_min, grid.lat_max,
                                   grid.lon_min, grid.lon_max, 40)
    im2 = ax2.imshow(hm_gapbm, origin="lower", cmap=CMAP_HEAT, aspect="auto",
                     extent=[grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max])
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_title("GAPBM Perturbed Heatmap", fontweight="bold")
    ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")

    # ── 行1列3: FixedDP热力图
    ax3 = fig.add_subplot(gs[0, 2])
    hm_fixed = points_to_heatmap(perturbed_fixed, grid.lat_min, grid.lat_max,
                                   grid.lon_min, grid.lon_max, 40)
    im3 = ax3.imshow(hm_fixed, origin="lower", cmap=CMAP_HEAT, aspect="auto",
                     extent=[grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max])
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    ax3.set_title("Fixed-ε DP Heatmap", fontweight="bold")
    ax3.set_xlabel("Longitude"); ax3.set_ylabel("Latitude")

    # ── 行2列1: 轨迹样例对比
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(min(4, len(original))):
        ax4.plot(original[i][:, 1], original[i][:, 0], "-", color="#2196F3", alpha=0.6, linewidth=1)
    ax4.set_title("Original Trajectories (sample)", fontweight="bold")
    ax4.set_xlabel("Longitude"); ax4.set_ylabel("Latitude")

    ax5 = fig.add_subplot(gs[1, 1])
    for i in range(min(4, len(perturbed_gapbm))):
        ax5.plot(perturbed_gapbm[i][:, 1], perturbed_gapbm[i][:, 0],
                 "-", color="#4CAF50", alpha=0.6, linewidth=1)
    ax5.set_title("GAPBM Perturbed (sample)", fontweight="bold")
    ax5.set_xlabel("Longitude"); ax5.set_ylabel("Latitude")

    # ── 行2列3: 指标条形图
    ax6 = fig.add_subplot(gs[1, 2])
    key_metrics = ["MAE(°)", "RMSE(°)", "TLR", "DDA(°)", "KL_Div"]
    methods_list = list(results.keys())
    x = np.arange(len(key_metrics))
    w = 0.35
    bar_colors = {"GAPBM": "#1976D2", "FixedDP": "#F57C00"}
    for j, m in enumerate(methods_list):
        vals = [results[m].get(k, 0) for k in key_metrics]
        ax6.bar(x + j * w - w / 2, vals, w,
                label=m, color=bar_colors.get(m, "#888"), alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels(key_metrics, rotation=25, ha="right", fontsize=8)
    ax6.set_title("Key Metrics Comparison", fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "GAPBM vs Fixed-ε DP: Comprehensive Summary Dashboard",
        fontsize=15, fontweight="bold", y=1.01
    )
    return _save(fig, filename)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import generate_synthetic_geolife_data, get_trajectories_list
    from grid import build_spatial_grid, compute_sensitivity_scores
    from gapbm import allocate_adaptive_budget, perturb_trajectories_gapbm, perturb_trajectories_fixed_dp
    from metrics import evaluate_all

    df = generate_synthetic_geolife_data(30)
    trajs = get_trajectories_list(df)
    grid = build_spatial_grid(df, 20, 20)
    grid = compute_sensitivity_scores(grid)
    grid = allocate_adaptive_budget(grid, 1.0)
    p_gapbm = perturb_trajectories_gapbm(trajs, grid)
    p_fixed = perturb_trajectories_fixed_dp(trajs, 1.0, grid)
    results = evaluate_all(trajs, p_gapbm, p_fixed,
                           grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max)
    plot_summary_dashboard(trajs, p_gapbm, p_fixed, grid, results)
    print("可视化测试完成")
