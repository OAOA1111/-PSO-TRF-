import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
import os
import glob

from PSO_TRS_Optimize import PSO_IRF

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'


# ==========================================
# 1. 基础函数 (与之前一致)
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
def process_single_file(file_path, output_dir, bounds):
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

    num_peaks = len(bounds) // 4
    
    def residual(params):
        return multi_voigt(x_data, *params) - y_data

    # --- PSO 全局搜索 + TRF 局部精修 ---
    optimizer = PSO_IRF(
        func=residual,
        bounds=bounds,
        args=(),
        num_particles=100,
        max_iter=200,
    )
    fit_result = optimizer.fit(verbose=False)
    final_params = fit_result["trf_params"]

    # --- 计算结果并导出 ---
    y_fit = multi_voigt(x_data, *final_params)
    
    # 1. 导出拟合曲线数据
    fit_df = pd.DataFrame({
        'Wavenumber': x_data,
        'Original_Intensity': y_data,
        'Fitted_Intensity': y_fit
    })
    fit_csv_path = os.path.join(output_dir, f"{base_name}_fitted_curve.csv")
    fit_df.to_csv(fit_csv_path, index=False)

    # 2. 导出各峰参数
    peak_info = []
    for i in range(num_peaks):
        idx = i * 4
        A, c, s, g = final_params[idx:idx+4]
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
    plt.scatter(x_data, y_data, s=10, label='Original Data', color='black', alpha=0.5)
    plt.plot(x_data, y_fit, label='Cumulative Fit', color='red', linewidth=2)
    
    for i in range(num_peaks):
        idx = i * 4
        c_i = final_params[idx+1]
        single_y = single_voigt(x_data, *final_params[idx:idx+4])
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

    # 开始循环遍历文件
    for file_path in file_list:
        process_single_file(file_path, output_folder, bounds)
        
    print("\n 所有批量任务执行完毕！")

if __name__ == "__main__":
    main()