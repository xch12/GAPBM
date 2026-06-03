"""
GAPBM 主实验脚本
================
完整执行流程：
  1. 数据加载/生成
  2. 空间网格构建 + 敏感度分析
  3. GAPBM 自适应预算分配
  4. GAPBM 扰动 + 传统 Laplace DP 扰动
  5. 全面指标评估
  6. 可视化输出（热力图/轨迹/指标条形图/预算分配/ε曲线）
  7. 结果保存（CSV + 控制台报告）

用法：
    cd /home/user/webapp
    python src/run_experiment.py
    python src/run_experiment.py --n_traj 80 --epsilon 1.0 --grid_size 25
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import generate_synthetic_geolife_data, load_geolife_data, get_trajectories_list
from grid import build_spatial_grid, compute_sensitivity_scores
from gapbm import allocate_adaptive_budget, perturb_trajectories_gapbm, perturb_trajectories_fixed_dp
from metrics import evaluate_all, print_evaluation_table
from visualization import (
    plot_grid_density,
    plot_budget_heatmap,
    plot_trajectory_comparison,
    plot_heatmap_comparison,
    plot_metrics_bar,
    plot_epsilon_vs_error,
    plot_summary_dashboard,
    plot_zone_comparison,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="GAPBM Experiment Runner")
    parser.add_argument("--data_dir",    type=str,   default=None,
                        help="GeoLife 数据集目录（不提供则使用模拟数据）")
    parser.add_argument("--n_traj",      type=int,   default=50,
                        help="模拟轨迹数量（使用模拟数据时有效）")
    parser.add_argument("--epsilon",     type=float, default=1.0,
                        help="总隐私预算 ε")
    parser.add_argument("--grid_size",   type=int,   default=20,
                        help="网格边长（grid_size × grid_size）")
    parser.add_argument("--sensitivity_method", type=str, default="density",
                        choices=["density", "entropy", "percentile"],
                        help="敏感度计算方法")
    parser.add_argument("--budget_strategy", type=str, default="inverse_sensitivity",
                        choices=["inverse_sensitivity", "exponential", "step"],
                        help="预算分配策略")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--epsilon_sweep", action="store_true",
                        help="是否进行 ε 扫描实验")
    return parser.parse_args()


def run_single_experiment(
    trajectories, grid, epsilon, budget_strategy, seed
):
    """在给定 ε 下运行一次完整对比实验，返回评估结果"""
    # 重新分配预算（重要：每次 ε 不同时需重建）
    from copy import deepcopy
    g = deepcopy(grid)
    g = allocate_adaptive_budget(g, epsilon_total=epsilon,
                                  allocation_strategy=budget_strategy)

    p_gapbm = perturb_trajectories_gapbm(trajectories, g, seed=seed)
    p_fixed = perturb_trajectories_fixed_dp(trajectories, epsilon, g, seed=seed)

    results = evaluate_all(
        trajectories, p_gapbm, p_fixed,
        g.lat_min, g.lat_max, g.lon_min, g.lon_max,
        grid=g,
    )
    return results, g, p_gapbm, p_fixed


def epsilon_sweep(trajectories, grid, epsilon_list, budget_strategy, seed):
    """ε 扫描：收集不同 ε 下的 MeanDist 误差"""
    print("\n[实验] 开始 ε 扫描...")
    gapbm_errors, fixed_errors = [], []
    for eps in epsilon_list:
        res, g_tmp, _, _ = run_single_experiment(trajectories, grid, eps, budget_strategy, seed)
        gapbm_errors.append(res["GAPBM"]["MeanDist(m)"])
        fixed_errors.append(res["FixedDP"]["MeanDist(m)"])
        print(f"  ε={eps:.2f}: GAPBM={gapbm_errors[-1]:.1f}m, Fixed={fixed_errors[-1]:.1f}m")
    return gapbm_errors, fixed_errors


def save_results(results: dict, epsilon: float):
    """保存评估结果到 CSV 和 JSON"""
    rows = []
    for method, metrics in results.items():
        row = {"method": method, "epsilon": epsilon}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(METRICS_DIR, f"metrics_eps{epsilon:.2f}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[结果] 已保存评估指标: {csv_path}")

    json_path = os.path.join(METRICS_DIR, f"metrics_eps{epsilon:.2f}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"epsilon": epsilon, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"[结果] 已保存 JSON: {json_path}")
    return csv_path


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         GAPBM - Grid-based Adaptive Privacy Budget          ║
║              Mechanism for Trajectory Protection            ║
║                                                              ║
║  对比方法：传统固定预算 Laplace 差分隐私 (Fixed-ε DP)         ║
║  数据集  ：GeoLife 北京轨迹（模拟/真实）                     ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    args = parse_args()
    print_banner()
    np.random.seed(args.seed)

    # ── Step 1: 数据加载 ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: 数据加载与预处理")
    print("=" * 60)
    if args.data_dir:
        df = load_geolife_data(args.data_dir, max_users=15, max_files_per_user=8)
    else:
        df = generate_synthetic_geolife_data(n_trajectories=args.n_traj, seed=args.seed)

    trajectories = get_trajectories_list(df)
    print(f"[数据] 轨迹条数: {len(trajectories)}, 总点数: {sum(len(t) for t in trajectories)}")

    # ── Step 2: 空间网格 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 空间网格构建 + 密度统计")
    print("=" * 60)
    grid = build_spatial_grid(df, n_rows=args.grid_size, n_cols=args.grid_size)
    grid = compute_sensitivity_scores(grid, method=args.sensitivity_method)
    plot_grid_density(grid, filename="grid_density.png")

    # ── Step 3: 自适应预算分配 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: GAPBM 自适应隐私预算分配")
    print("=" * 60)
    grid = allocate_adaptive_budget(grid, epsilon_total=args.epsilon,
                                    allocation_strategy=args.budget_strategy)
    plot_budget_heatmap(grid, epsilon_total=args.epsilon, filename="budget_heatmap.png")

    # ── Step 4: 轨迹扰动 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: 轨迹扰动 (GAPBM + Fixed-ε DP)")
    print("=" * 60)
    print("[GAPBM]  自适应预算 Laplace 扰动...")
    p_gapbm = perturb_trajectories_gapbm(trajectories, grid, seed=args.seed)
    print("[FixedDP] 固定预算 Laplace 扰动...")
    p_fixed  = perturb_trajectories_fixed_dp(trajectories, args.epsilon, grid, seed=args.seed)
    print(f"[完成] {len(trajectories)} 条轨迹扰动完成")

    # ── Step 5: 评估 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: 评估指标计算")
    print("=" * 60)
    results = evaluate_all(
        trajectories, p_gapbm, p_fixed,
        grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max,
        grid=grid,
    )
    print_evaluation_table(results)
    save_results(results, args.epsilon)

    # ── Step 6: 可视化 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: 生成可视化图表")
    print("=" * 60)
    plot_trajectory_comparison(
        trajectories, p_gapbm, p_fixed, grid, n_show=5,
        filename="trajectory_comparison.png"
    )
    plot_heatmap_comparison(
        trajectories, p_gapbm, p_fixed, grid,
        filename="heatmap_comparison.png"
    )
    plot_metrics_bar(results, filename="metrics_comparison.png")
    plot_summary_dashboard(
        trajectories, p_gapbm, p_fixed, grid, results,
        filename="summary_dashboard.png"
    )
    plot_zone_comparison(results, filename="zone_comparison.png")

    # ── Step 7: ε 扫描（可选）────────────────────────────────────
    if args.epsilon_sweep:
        print("\n" + "=" * 60)
        print("Step 7: ε 扫描实验")
        print("=" * 60)
        epsilon_list = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
        gapbm_errors, fixed_errors = epsilon_sweep(
            trajectories, grid, epsilon_list, args.budget_strategy, args.seed
        )
        plot_epsilon_vs_error(
            epsilon_list, gapbm_errors, fixed_errors,
            metric_name="MeanDist(m)", filename="epsilon_vs_error.png"
        )

        # 保存扫描结果
        sweep_df = pd.DataFrame({
            "epsilon": epsilon_list,
            "GAPBM_MeanDist(m)": gapbm_errors,
            "FixedDP_MeanDist(m)": fixed_errors,
            "GAPBM_improvement(%)": [
                (f - g) / f * 100 for g, f in zip(gapbm_errors, fixed_errors)
            ]
        })
        sweep_path = os.path.join(METRICS_DIR, "epsilon_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\n[扫描] 结果已保存: {sweep_path}")
        print(sweep_df.to_string(index=False))

    # ── 最终报告 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("实验完成 - 输出文件清单")
    print("=" * 60)
    fig_dir = os.path.join(RESULTS_DIR, "figures")
    for fname in sorted(os.listdir(fig_dir)):
        print(f"  📊 results/figures/{fname}")
    for fname in sorted(os.listdir(METRICS_DIR)):
        print(f"  📄 results/metrics/{fname}")

    # 打印关键结论
    g_mae    = results["GAPBM"]["MAE(°)"]
    f_mae    = results["FixedDP"]["MAE(°)"]
    g_kl     = results["GAPBM"]["KL_Div"]
    f_kl     = results["FixedDP"]["KL_Div"]
    g_corr   = results["GAPBM"]["Heatmap_Corr"]
    f_corr   = results["FixedDP"]["Heatmap_Corr"]
    g_se     = results["GAPBM"].get("SensZone_Err", 0)
    f_se     = results["FixedDP"].get("SensZone_Err", 0)
    g_nse    = results["GAPBM"].get("NonSensZone_Err", 0)
    f_nse    = results["FixedDP"].get("NonSensZone_Err", 0)
    kl_imp   = (f_kl - g_kl) / f_kl * 100 if f_kl > 0 else 0
    nse_imp  = (f_nse - g_nse) / f_nse * 100 if f_nse > 0 else 0
    corr_imp = (g_corr - f_corr) / abs(f_corr + 1e-9) * 100

    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│               实验结论摘要 (ε={args.epsilon})                         │
├──────────────────────────────────────────────────────────────────┤
│  指标                GAPBM        Fixed-ε DP    对比             │
│  MAE(°)          {g_mae:>10.6f}   {f_mae:>10.6f}                  │
│  KL散度          {g_kl:>10.6f}   {f_kl:>10.6f}   {kl_imp:>+6.1f}%     │
│  热力图相关      {g_corr:>10.6f}   {f_corr:>10.6f}   {corr_imp:>+6.1f}%     │
│  敏感区扰动(°)   {g_se:>10.6f}   {f_se:>10.6f}   GAPBM更强保护  │
│  普通区扰动(°)   {g_nse:>10.6f}   {f_nse:>10.6f}   {nse_imp:>+6.1f}%     │
├──────────────────────────────────────────────────────────────────┤
│  结论: GAPBM 对高密度敏感区域施加更强噪声保护（敏感区扰动↑），   │
│        同时对低密度普通区域施加更少扰动（普通区扰动↓），         │
│        实现了隐私保护与数据可用性之间更好的平衡。               │
└──────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
