'''
arPLS 基线估计（Asymmetrically Reweighted Penalized Least Squares, Baek et al. 2015）

通过带惩罚项的加权最小二乘迭代估计平滑基线：峰处权重自动→0、基线处权重→1，
从而在扣除基线时不影响真实信号。无需手动选择基线区域，适合批量光谱处理。
'''
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def arpls(y, lam=1e6, ratio=1e-6, max_iter=20):
    '''
    arPLS 基线估计。

    参数
    ----
    y : array-like
        输入光谱强度。
    lam : float
        平滑度惩罚系数。越大基线越刚（更平滑、不易跟随峰）；
        越小越柔（可能扣到峰上）。DRIFTS 常用 1e5~1e7。
    ratio : float
        权重收敛阈值，相邻两次权重相对变化小于该值则停止。
    max_iter : int
        最大迭代次数。

    返回
    ----
    z : np.ndarray
        估计的基线，与 y 等长。
    '''
    y = np.asarray(y, dtype=float)
    N = len(y)

    # 二阶差分算子 D, shape (N-2, N)：惩罚基线的二阶非平滑（弯曲）
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N), format='csc')
    H = lam * (D.T @ D)

    w = np.ones(N)
    z = y.copy()
    for _ in range(max_iter):
        # 解 (W + lam*D'D) z = w * y
        W = sparse.diags(w, 0, shape=(N, N), format='csc')
        z = spsolve(W + H, w * y)
        d = y - z

        # 用负残差（基线以下视为噪声）统计量驱动权重
        dn = d[d < 0]
        if dn.size == 0:
            break
        m, s = np.mean(dn), np.std(dn)
        if s < 1e-12:
            s = 1e-12

        # 峰处（d 大正）→ wt→0，基线处（d 小/负）→ wt→1
        wt = 1.0 / (1.0 + np.exp(2 * (d - (2 * s - m)) / s))

        if np.linalg.norm(w - wt) / (np.linalg.norm(w) + 1e-12) < ratio:
            w = wt
            break
        w = wt

    return z


def poly_baseline(x, coeffs):
    '''
    多项式基线：y = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...（升幂）。

    为改善数值条件，建议传入归一化的 x
    （如 xn = (x - x.mean()) / x.std()），使各阶系数量级相近，
    便于在联合拟合中设置统一的边界。
    '''
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        y += c * (x ** i)
    return y

