# 光谱多峰拟合算法（PSO + TRF 混合优化）

基于 **粒子群优化（PSO）全局搜索** 与 **Trust Region Reflective（TRF）局部精修** 相结合的混合算法，对光谱数据进行多峰 Voigt 函数拟合。专为批量处理红外 / 拉曼光谱而设计，可自动输出拟合曲线、各峰参数与高分辨率图谱。

---

## ✨ 特性

- **混合优化策略**：用 PSO 进行全局搜索跳出局部最优，再以 TRF 在边界约束下做高精度局部精修，兼顾鲁棒性与精度。
- **多峰 Voigt 拟合**：Voigt 函数为高斯与洛伦兹的卷积，能更真实地描述光谱峰形（同时反映仪器展宽与寿命展宽）。
- **批量自动处理**：自动遍历输入文件夹内所有 CSV，逐文件完成拟合并导出结果，全程无需人工干预。
- **自动 FWHM 计算**：基于 Olivero–Longbothum 近似公式由 σ、γ 反推 Voigt 半高全宽。
- **基线校正（四种方法可选）**：arPLS 自动基线 / 多项式联合拟合 / 线性联合拟合 / 关闭，一行配置切换，避免基线漂移被错误吸收进峰形（面积、峰宽失真）。详见下文。
- **一键出图**：自动绘制原始数据、基线、校正数据、累积拟合曲线与各分峰，600 dpi 保存，可直接用于论文。

---

## 🧠 算法原理

### 1. PSO 全局搜索

粒子群优化（Particle Swarm Optimization）通过模拟鸟群觅食行为寻找最优解。每个粒子代表一组候选参数（即各峰的 Amplitude、Center、Sigma、Gamma）。

迭代更新规则：

$$v_{i}^{t+1} = w \cdot v_{i}^{t} + c_1 r_1 (p_{best,i} - x_i) + c_2 r_2 (g_{best} - x_i)$$

$$x_{i}^{t+1} = x_i + v_{i}^{t+1}$$

其中：
- $w$ 为惯性权重，本算法采用**线性递减策略**（`w_max → w_min`），前期大权重利于全局探索，后期小权重利于局部开发；
- $c_1$、$c_2$ 为个体 / 群体学习因子；
- 粒子位置被截断（`np.clip`）约束在设定边界内，防止越界。

目标函数采用与最小二乘一致的定义：$L = 0.5 \sum r_i^2$（$r$ 为残差向量），不可行解返回 `inf`。

### 2. TRF 局部精修

PSO 收敛后，以其全局最优位置 `gbest` 作为初值，调用 `scipy.optimize.least_squares` 的 **Trust Region Reflective** 方法进行有边界约束的局部精修，收敛容差设为 `1e-10`，从而在 PSO 找到的解邻域内取得高精度结果。

### 3. Voigt 峰形

单个 Voigt 峰定义为：

$$V(x) = A \cdot \text{voigt\_profile}(x - c,\ \sigma,\ \gamma)$$

其中 $A$ 为面积（振幅）、$c$ 为峰中心、$\sigma$ 为高斯展宽、$\gamma$ 为洛伦兹展宽。多峰拟合为多个 Voigt 峰的线性叠加，**每个峰 4 个参数**。

Voigt 半高全宽（FWHM）由 Olivero–Longbothum 近似计算：

$$f_g = 2\sigma\sqrt{2\ln 2},\quad f_l = 2\gamma,\quad f_v = 0.5346\,f_l + \sqrt{0.2166\,f_l^2 + f_g^2}$$

---

## 📁 项目结构

```
光谱拟合算法/
├── PSO_TRS_Optimize.py   # PSO + TRF 混合优化器核心类 PSO_IRF
├── baseline.py            # 基线估计：arPLS 自动基线 + 多项式基线（poly/linear 联合拟合用）
├── 批量处理.py            # 批量光谱拟合脚本（基线校正 + 多峰 Voigt 拟合 + 导出 + 绘图）
├── input_spectra/         # 输入文件夹（放入待拟合 CSV，自动创建）
├── output_results/        # 输出文件夹（结果自动写入，自动创建）
└── README.md
```

---

## 📐 基线校正

DRIFTS / Kubelka-Munk 光谱常带弯曲与倾斜的基线。若不扣除，基线漂移会被错误吸收进峰形，导致面积与峰宽失真。本算法提供 **四种可切换的基线处理方式**，通过 `BASELINE_CFG["method"]` 选择：

| `method` | 说明 | 思路 | 适用场景 |
|----------|------|------|----------|
| `"arpls"` | arPLS 自动基线（**推荐**，默认） | 预处理：先估计并扣除基线，再对校正数据拟合峰 | 弯曲/倾斜基线、批量处理 |
| `"poly"` | 多项式基线联合拟合 | 基线系数并入参数向量，与峰参数一起由 PSO+TRF 优化 | 基线近似低阶多项式 |
| `"linear"` | 线性基线联合拟合 | 斜率+截距 2 参数与峰联合优化（即 `"poly"` 的 1 阶特例） | 近似平直/线性倾斜基线 |
| `"none"` | 不做基线校正 | 直接拟合原始数据 | 已扣除基线或无需校正 |

### arPLS（`"arpls"`）

Asymmetrically Reweighted Penalized Least Squares（Baek et al. 2015）：在带平滑惩罚的最小二乘框架下迭代求解基线 `z`，使 `(W + λDᵀD) z = w⊙y`；用负残差统计量驱动 Logistic 权重（峰处→0、基线处→1），只拟合基线不破坏信号；`scipy.sparse` 稀疏求解，长光谱也高效。完全自动、无需选区。

**`lam` 调参**：过小→基线太柔，可能"扣到峰上"削弱信号；过大→基线太刚，残留漂移。经验先取 `1e6`，扣除过度则上调一个量级，残留明显则下调（DRIFTS 常用 `1e5~1e7`）。

### 多项式 / 线性联合拟合（`"poly"` / `"linear"`）

将基线作为模型的一部分，与峰参数**一起优化**：

- 模型 = `多峰 Voigt + 多项式基线`，残差对**原始数据** `y_data` 计算；
- 基线系数追加在 `bounds` 末尾，参与 PSO 全局搜索 + TRF 精修；
- `x` 在内部做归一化（`(x-mean)/std`），使各阶系数量级相近、便于统一设界；
- `"linear"` 等价于 1 阶（斜率+截距，2 参数），`"poly"` 默认 2 阶（3 参数）。

> 物理一致性更好（基线在考虑峰形的情况下拟合），复用现有优化器；但全局多项式对强弯曲基线偏刚性，且增加搜索维度。**仅基线近低阶多项式时建议使用，否则优先 `arpls`。**

### 配置（`批量处理.py` 的 `main()` 内）

```python
BASELINE_CFG = {
    "method": "arpls",              # arpls / poly / linear / none
    # --- arpls 参数 ---
    "lam": 1e6,                     # 平滑度，越大越刚
    "ratio": 1e-6,                  # 权重收敛阈值
    "max_iter": 20,                 # 最大迭代次数
    # --- poly / linear 参数 ---
    "poly_degree": 2,               # 仅 "poly" 生效，多项式阶数
    "poly_bounds": (-100, 100),     # 各基线系数上下界（基于归一化 x）
}
```

切换方法只需改 `"method"` 一行。设 `"none"` 可完全关闭基线校正（输出中 `Baseline` 列全 0、`Corrected_Intensity` 与 `Original_Intensity` 相同）。各方法输出格式一致：CSV 均含 `Baseline` 与 `Corrected_Intensity` 列（见下文输出说明）。

---

## 🔧 依赖环境

| 依赖 | 用途 |
|------|------|
| `numpy` | 数值计算、向量化运算 |
| `pandas` | CSV 读写、结果整理 |
| `scipy` | `voigt_profile` 峰形函数、`least_squares`（TRF）优化 |
| `matplotlib` | 拟合结果绘图 |

安装：

```bash
pip install numpy pandas scipy matplotlib
```

---

## 🚀 快速开始

1. **放入数据**：将待拟合的光谱 CSV 文件放入 `input_spectra/` 文件夹（首次运行会自动创建）。

   - CSV 格式：**无表头**，两列以逗号分隔；
   - 第 1 列：波数（Wavenumber, cm⁻¹）；
   - 第 2 列：强度（如 Kubelka-Munk 单位）。

2. **运行批量处理**：

   ```bash
   python 批量处理.py
   ```

3. **查看结果**：所有输出写入 `output_results/`，每个输入文件生成三份结果（见下文）。

---

## 📤 输出说明

对每个输入文件 `{文件名}.csv`，输出以下三份文件：

| 文件 | 内容 |
|------|------|
| `{文件名}_fitted_curve.csv` | 波数、原始强度、基线（arPLS）、校正强度、拟合强度（用于二次绘图） |
| `{文件名}_peak_parameters.csv` | 各峰编号、中心、面积、高斯 σ、洛伦兹 γ、近似 FWHM（基于扣基线后数据） |
| `{文件名}_plot.png` | 高分辨率（600 dpi）拟合图：原始数据散点 + 基线（绿虚线）+ 校正数据散点 + 累积拟合曲线 + 各分峰虚线 |

---

## ⚙️ 参数配置

### PSO 优化参数（`批量处理.py` 中 `PSO_IRF(...)` 调用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_particles` | `100` | 粒子数量，越多搜索能力越强但越慢 |
| `max_iter` | `200` | PSO 最大迭代次数 |
| `w_min` / `w_max` | `0.1` / `0.7` | 惯性权重递减区间（线性递减） |
| `c1` / `c2` | `1.5` / `1.5` | 个体 / 群体学习因子 |
| `vmax` | `None` | 最大速度限制，`None` 表示不限制 |
| `seed` | `None` | 随机种子，设置后 PSO 结果可复现；`批量处理.py` 已设为 `42` |

### 拟合峰与边界（`批量处理.py` 中 `bounds` 列表）

`bounds` 按**每峰 4 个参数**顺序排列：`Amplitude、Center、Sigma、Gamma`，每个为 `(下限, 上限)` 元组。默认配置为 4 个峰：

```python
bounds = [
    # 峰 1
    (0, 30),        # Amplitude 面积
    (1550, 1570),   # Center 峰中心 (cm⁻¹)
    (6, 10),        # Sigma 高斯展宽
    (0, 10),        # Gamma 洛伦兹展宽
    # 峰 2
    (0, 30),
    (1575, 1590),
    (6, 12),
    (0, 10),
    # 峰 3
    (0, 30),
    (1605, 1615),
    (1, 13),
    (3, 10),
    # 峰 4
    (0, 30),
    (1655, 1670),
    (1, 15),
    (1, 15),
]
```

> 修改峰数：直接增删 `bounds` 中以 4 个为一组的峰配置即可，`num_peaks = len(bounds) // 4` 会自动适配。建议根据实际光谱的吸收 / 散射峰位置合理设置 `Center` 边界，以保证拟合物理意义。

---

## 🧩 核心类 `PSO_IRF` 接口

`PSO_TRS_Optimize.py` 中的 `PSO_IRF` 是通用优化器，可独立用于任意带边界约束的最小二乘问题。

```python
from PSO_TRS_Optimize import PSO_IRF

# 定义残差函数：residual(params, *args) -> 残差向量
def residual(params, *args):
    return model(params) - y_data

optimizer = PSO_IRF(
    func=residual,
    bounds=bounds,        # [(low, high), ...]
    args=(),
    num_particles=100,
    max_iter=200,
    seed=42,
)

# 一键完成 PSO 全局搜索 + TRF 局部精修
result = optimizer.fit(verbose=False)

final_params = result["trf_params"]   # 最终参数
final_cost   = result["trf_cost"]    # 最终代价
```

`fit()` 返回字典字段：

| 字段 | 说明 |
|------|------|
| `pso_params` / `pso_loss` | PSO 全局最优位置与目标值 |
| `trf_params` / `trf_cost` | TRF 精修后的最终参数与代价 |
| `trf_success` / `trf_message` | TRF 收敛状态与信息 |
| `trf_result` | `scipy` 原始 `OptimizeResult` 对象 |

---

## 📌 注意事项

- **坐标系**：绘图时 X 轴（波数）自动反转（`invert_xaxis`），符合红外 / 拉曼光谱习惯。
- **内存管理**：批量绘图后调用 `plt.close()` 关闭图窗，避免大量文件时内存泄漏。
- **拟合质量**：若拟合不佳，可增大 `num_particles` / `max_iter`，或收紧 / 调整各峰 `Center` 边界。
- **结果复现**：设置 `seed` 可使 PSO 初始化与随机过程可复现。
- **峰数一致性**：所有输入文件共用同一套 `bounds`，请确保输入光谱的峰位相近；若样品差异大，建议按组分批处理并各自设置边界。

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源，可自由使用、修改与分发，请保留原始版权与许可声明。
