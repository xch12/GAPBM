# GAPBM: Grid-based Adaptive Privacy Budget Mechanism

> **轨迹隐私保护 | 自适应差分隐私 | GeoLife 北京轨迹**

---

## 研究背景

传统差分隐私方法对所有轨迹点使用**固定隐私预算**，但现实中不同位置的敏感程度不同：
- 高密度区域（天安门、中关村等热点）→ 需要更强的隐私保护
- 低密度普通区域 → 不必要的强扰动会降低数据可用性

**GAPBM** 通过基于空间网格的自适应预算分配，解决固定预算方法在隐私保护与数据可用性之间难以平衡的问题。

---

## 方法概述

```
GeoLife轨迹数据
      │
      ▼
┌─────────────────┐
│  空间网格划分    │  将北京区域划分为 N×N 网格
│  Grid Building  │  统计各网格内轨迹点密度
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  敏感度计算      │  density / entropy / percentile
│  Sensitivity    │  高密度 → 高敏感度分数 S(i,j)∈[0,1]
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  自适应预算分配 (GAPBM核心)      │
│  ε(i,j) = ε_total × w(i,j)     │
│  w(i,j) = 1 - S(i,j)           │  ← 反敏感度权重
│  高敏感区 → 小ε → 强保护        │
│  低敏感区 → 大ε → 弱扰动        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Laplace 扰动   │  scale = Δf / ε(i,j)
│  Perturbation   │  p' = p + Laplace(0, scale)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  评估 & 对比     │  vs 传统固定预算 Laplace DP
│  Evaluation     │  MAE/RMSE/热力图/区域分析
└─────────────────┘
```

---

## 项目结构

```
GAPBM/
├── src/
│   ├── data_loader.py      # GeoLife 数据加载 & 模拟数据生成
│   ├── grid.py             # 空间网格划分 & 敏感度计算
│   ├── gapbm.py            # GAPBM核心：自适应预算分配 + Laplace扰动
│   ├── metrics.py          # 评估指标（MAE/RMSE/KL散度/区域分析）
│   ├── visualization.py    # 可视化（热力图/轨迹/指标/预算分配）
│   └── run_experiment.py   # 主实验脚本
├── data/
│   ├── raw/                # 原始 GeoLife 数据（.plt 文件）
│   └── processed/          # 预处理后数据
├── results/
│   ├── figures/            # 实验图表输出
│   └── metrics/            # 评估指标 CSV/JSON
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验（使用模拟数据）

```bash
cd /path/to/GAPBM
python src/run_experiment.py --n_traj 50 --epsilon 1.0 --grid_size 20 --epsilon_sweep
```

### 3. 使用真实 GeoLife 数据

下载 [Microsoft GeoLife Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52367)，解压后：

```bash
python src/run_experiment.py \
    --data_dir /path/to/GeoLife/Data \
    --epsilon 1.0 \
    --grid_size 25 \
    --epsilon_sweep
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_traj` | 50 | 模拟轨迹数量 |
| `--epsilon` | 1.0 | 总隐私预算 ε |
| `--grid_size` | 20 | 网格边长（N×N） |
| `--sensitivity_method` | density | 敏感度计算方法 `density/entropy/percentile` |
| `--budget_strategy` | inverse_sensitivity | 预算分配策略 `inverse_sensitivity/exponential/step` |
| `--epsilon_sweep` | False | 是否进行 ε 扫描对比实验 |
| `--seed` | 42 | 随机种子 |

---

## 评估指标

| 指标 | 说明 | 越好 |
|------|------|------|
| MAE(°) | 逐点平均绝对误差 | 小 |
| RMSE(°) | 均方根误差 | 小 |
| MeanDist(m) | 平均 Haversine 距离 | 小 |
| TLR | 轨迹长度比偏差 | 小 |
| DDA(°) | 方向偏差角 | 小 |
| KL_Div | 热力图 KL 散度 | 小 |
| Heatmap_Corr | 热力图 Pearson 相关 | 大 |
| **SensZone_Err** | **敏感区扰动幅度** | **GAPBM更大（更强保护）** |
| **NonSensZone_Err** | **普通区扰动幅度** | **GAPBM更小（更好可用性）** |

---

## 输出图表

| 文件 | 内容 |
|------|------|
| `grid_density.png` | 空间网格密度热力图 |
| `budget_heatmap.png` | 敏感度图 + 自适应预算分配图 |
| `trajectory_comparison.png` | 轨迹三列对比（原始/GAPBM/FixedDP） |
| `heatmap_comparison.png` | 热力图分布三列对比 |
| `metrics_comparison.png` | 各指标条形图对比 |
| `zone_comparison.png` | 敏感区/普通区扰动分析 |
| `epsilon_vs_error.png` | ε 扫描误差曲线 |
| `summary_dashboard.png` | 综合摘要仪表板 |

---

## 核心算法

### 自适应预算分配公式

$$\varepsilon(i,j) = \varepsilon_{total} \cdot \frac{w(i,j)}{\bar{w}}$$

其中反敏感度权重：

$$w(i,j) = 1 - S(i,j), \quad S(i,j) \in [0,1]$$

- $S(i,j)$ 越大（高密度热点）→ $w(i,j)$ 越小 → $\varepsilon(i,j)$ 越小 → **噪声更大 = 更强隐私保护**
- $S(i,j)$ 越小（低密度区域）→ $w(i,j)$ 越大 → $\varepsilon(i,j)$ 越大 → **噪声更小 = 更好数据可用性**

### Laplace 扰动

$$p'_{lat} = p_{lat} + \text{Lap}\left(0, \frac{\Delta f_{lat}}{\varepsilon(i,j)}\right)$$

$$p'_{lon} = p_{lon} + \text{Lap}\left(0, \frac{\Delta f_{lon}}{\varepsilon(i,j)}\right)$$

---

## 关键结论

GAPBM 相比传统固定预算 Laplace DP：

1. **敏感区域保护更强**：高密度热点区扰动幅度更大，有效防止位置泄露
2. **普通区域可用性更好**：低密度区域扰动幅度更小，保留更多轨迹语义
3. **热力图分布保持性更高**：Heatmap_Corr 更大，整体分布特征保留更完整
4. **隐私预算自适应分配**：根据轨迹密度动态调整，实现细粒度的隐私控制

---

## 引用

```bibtex
@misc{gapbm2025,
  title={GAPBM: Grid-based Adaptive Privacy Budget Mechanism for Trajectory Protection},
  author={xch12},
  year={2025},
  note={Based on GeoLife Beijing Trajectory Dataset}
}
```

---

## 数据集

Microsoft Research GeoLife GPS Trajectories  
- **论文**: Yu Zheng et al., "GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory"  
- **下载**: https://www.microsoft.com/en-us/download/details.aspx?id=52367
