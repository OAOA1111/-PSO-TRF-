import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
import os
import glob

from PSO_TRS_Optimize import PSO_IRF
from baseline import arpls, poly_baseline

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'


# ==========================================
# 1. 基础函数
# ==========================================
def single_voigt(x, amplitude, center, sigma, gamma):
    return amplitude * voigt_profile(x - center, sigma, gamma)

def multi_voigt(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        y += single_voigt(x, params[i], params[i+1], params[i+2], params[i+3])
    return y

def calculate_voigt_fwhm(sigma, gamma):
    f_g = 2 * sigma * np.sqrt(2 * np.log(2))
    f_l = 2 * gamma
    f_v = 0.5346 * f_l + np.sqrt(0.2166 * f_l**2 + f_g**2)
    return f_v

# ==========================================
# 2. 单个文件处理
# ==========================================
def process_single_file(file_path, output_dir, bounds, baseline_cfg):
    # 提取纯文件名（不含扩展名），用于命名输出文件
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n[{base_name}] 开始处理...")

    try:
        df = pd.read_csv(file_path, header=None)
        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values
    except Exception as e:
        print(f"[{base_name}] 读取失败跳过，错误: {e}")
        return

    # --- 基线校正方法选择 ---
    # method: "arpls"  arPLS 预处理（推荐，自动、对弯曲基线鲁棒）
    #         "poly"   多项式基线与峰联合拟合（poly_degree 阶）
    #         "linear" 线性基线与峰联合拟合（斜率+截距）
    #         "none"   不做基线校正
    method = baseline_cfg.get("method", "arpls")
    num_peaks = len(bounds) // 4

    if method == "arpls":
        # arPLS 预处理：先扣基线，再对校正数据拟合峰
        baseline = arpls(y_data,
                         lam=baseline_cfg.get("lam", 1e6),
                         ratio=baseline_cfg.get("ratio", 1e-6),
                         max_iter=baseline_cfg.get("max_iter", 20))
        y_target = y_data - baseline
        fit_bounds = bounds
        def model(params):
            return multi_voigt(x_data, *params)
    elif method in ("poly", "linear"):
        # 多项式/线性基线联合拟合：基线系数与峰参数一起优化
        degree = baseline_cfg.get("poly_degree", 2) if method == "poly" else 1
        b_lo, b_hi = baseline_cfg.get("poly_bounds", (-100, 100))
        # 归一化 x，使各阶基线系数量级相近，便于统一设界
        x_mean, x_std = float(np.mean(x_data)), float(np.std(x_data))
        if x_std < 1e-12:
            x_std = 1.0
        xn = (x_data - x_mean) / x_std
        n_peak_params = len(bounds)
        fit_bounds = list(bounds) + [(b_lo, b_hi)] * (degree + 1)
        y_target = y_data
        def model(params):
            peak_params = params[:n_peak_params]
            poly_params = params[n_peak_params:]
            return multi_voigt(x_data, *peak_params) + poly_baseline(xn, poly_params)
    elif method == "none":
        baseline = np.zeros_like(y_data)
        y_target = y_data
        fit_bounds = bounds
        def model(params):
            return multi_voigt(x_data, *params)
    else:
        raise ValueError(f"未知基线方法: {method}（可选: arpls / poly / linear / none）")

    def residual(params):
        return model(params) - y_target

    # --- PSO 全局搜索 + TRF 局部精修 ---
    optimizer = PSO_IRF(
        func=residual,
        bounds=fit_bounds,
        args=(),
        num_particles=100,
        max_iter=200,
        seed=42,
    )
    fit_result = optimizer.fit(verbose=False)
    final_params = fit_result["trf_params"]

    # --- 分离峰参数与基线 ---
    if method in ("poly", "linear"):
        peak_params = final_params[:n_peak_params]
        baseline = poly_baseline(xn, final_params[n_peak_params:])
    else:
        peak_params = final_params

    # --- 计算结果并导出 ---
    y_fit = multi_voigt(x_data, *peak_params)
    y_corr = y_data - baseline

    # 1. 导出拟合曲线数据
    fit_df = pd.DataFrame({
        'Wavenumber': x_data,
        'Original_Intensity': y_data,
        'Baseline': baseline,
        'Corrected_Intensity': y_corr,
        'Fitted_Intensity': y_fit
    })
    fit_csv_path = os.path.join(output_dir, f"{base_name}_fitted_curve.csv")
    fit_df.to_csv(fit_csv_path, index=False)

    # 2. 导出各峰参数
    peak_info = []
    for i in range(num_peaks):
        idx = i * 4
        A, c, s, g = peak_params[idx:idx+4]
        fwhm = calculate_voigt_fwhm(s, g)
        peak_info.append({
            'Peak_ID': i + 1,
            'Center': round(c, 4),
            'Amplitude (Area)': round(A, 4),
            'Sigma (Gaussian)': round(s, 4),
            'Gamma (Lorentzian)': round(g, 4),
            'FWHM (Approx)': round(fwhm, 4)
        })
    info_df = pd.DataFrame(peak_info)
    info_csv_path = os.path.join(output_dir, f"{base_name}_peak_parameters.csv")
    info_df.to_csv(info_csv_path, index=False)

    # 3. 自动绘图并保存为图片 (不显示界面)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, s=10, color='gray', alpha=0.3, label='Original Data')
    plt.plot(x_data, baseline, '--', color='green', linewidth=1.5, label='Baseline (arPLS)')
    plt.scatter(x_data, y_corr, s=10, color='black', alpha=0.5, label='Corrected Data')
    plt.plot(x_data, y_fit, color='red', linewidth=2, label='Cumulative Fit')
    
    for i in range(num_peaks):
        idx = i * 4
        c_i = peak_params[idx+1]
        single_y = single_voigt(x_data, *peak_params[idx:idx+4])
        plt.plot(x_data, single_y, '--', label = f"Peak {c_i:.1f} (cm$^{{-1}}$)")

    plt.xlabel('Wavenumber (cm$^{{-1}}$)', fontsize = 20)
    plt.ylabel('Kubelka- Munk units', fontsize = 20)
    plt.gca().invert_xaxis()
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend()
    plt.tight_layout()
    
    img_path = os.path.join(output_dir, f"{base_name}_plot.png")
    plt.savefig(img_path, dpi=600)
    plt.close() # 必须关闭，否则批量处理会导致内存泄漏

    print(f"[{base_name}] 处理完成！结果已存入输出文件夹。")

    

# ==========================================
# 3. 批量处理
# ==========================================
def main():
    # --- 文件夹配置 ---
    # 以脚本所在目录为基准，避免工作目录不一致导致路径错误
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "input_spectra")
    output_folder = os.path.join(script_dir, "output_results")

    # 如果文件夹不存在，自动创建
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"已创建 '{input_folder}' 文件夹，请将您的 CSV 文件放入其中，然后重新运行代码。")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有 csv 文件
    search_pattern = os.path.join(input_folder, "*.csv")
    file_list = glob.glob(search_pattern)

    if not file_list:
        print(f"在 '{input_folder}' 中没有找到 CSV 文件，请检查路径。")
        return

    print(f"找到 {len(file_list)} 个文件，准备开始批量拟合...")

    # --- 统一设置初始参数边界 ---
    bounds = [
        # 峰 1 的边界
         (0, 30),         # Amplitude 面积
        (1550, 1570),    # Center 峰中心位置
        (6, 10),         # Sigma (高斯展宽)
        (0,10),         # Gamma (洛伦兹展宽)
        # 峰 2 的边界
        (0, 30),         # Amplitude 面积
        (1575, 1590),    # Center 峰中心位置
        (6, 12),         # Sigma
        (0, 10),         # Gamma
        #峰 3 的边界
        (0, 30),         # Amplitude 面积
        (1605, 1615),    # Center 峰中心位置
        (1, 13),         # Sigma
        (3, 10),         # Gamma
        #峰 4 的边界
        (0, 30),         # Amplitude 面积
        (1655, 1670),    # Center 峰中心位置
        (1, 15),         # Sigma
        (1, 15),         # Gamma
    ]

    # --- 基线校正配置 ---
    # method 可选:
    #   "arpls"  arPLS 预处理（推荐，自动、对弯曲/倾斜基线鲁棒）
    #   "poly"   多项式基线与峰联合拟合（poly_degree 阶）
    #   "linear" 线性基线与峰联合拟合（斜率+截距）
    #   "none"   不做基线校正
    BASELINE_CFG = {
        "method": "arpls",
        # --- arpls 参数 ---
        "lam": 1e6,          # 平滑度，越大越刚；DRIFTS 常用 1e5~1e7
        "ratio": 1e-6,       # 权重收敛阈值
        "max_iter": 20,      # 最大迭代次数
        # --- poly / linear 参数 ---
        "poly_degree": 2,            # 仅 "poly" 生效，多项式阶数（2=二次）
        "poly_bounds": (-100, 100),   # 各基线系数上下界（基于归一化 x）
    }

    # 开始循环遍历文件
    for file_path in file_list:
        process_single_file(file_path, output_folder, bounds, BASELINE_CFG)
        
    print("\n 所有批量任务执行完毕！")

if __name__ == "__main__":
    main()