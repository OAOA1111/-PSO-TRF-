'''
改进的PSO与有边界约束的TRF混合算法
'''
import numpy as np
from scipy.optimize import least_squares
class PSO_IRF:
    def __init__(self, func, bounds, args=(), num_particles=100, max_iter=100, w_min=0.1, w_max=0.7, c1=1.5, c2=1.5,
                 vmax=None, seed=None, trf_options=None):
        '''
        func: 残差函数，形式是 residual_func(params, *args) -> residual_vector， 例如：return y_pred - y_data
        bounds: 每个变量的上下界，例如 [(-10, 10), (-10, 10)]
        args:传给func的额外参数
        num_particles: 粒子数量
        max_iter: 最大迭代次数
        w: 惯性权重
        c1: 个体学习因子
        c2: 群体学习因子
        vmax: 最大速度限制，可以是 None
        mode: "min" 表示最小化，"max" 表示最大化,默认最小化
        seed: 随机种子
        trf_options:传给scipy.optimize.least_squares的额外参数
        '''

        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.args = args
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_min = w_min
        self.w_max = w_max
        self.c1 = c1
        self.c2 = c2
        self.vmax = vmax
        self.trf_options = trf_options if trf_options is not None else {}

        self.dim = len(bounds)
        self.rng = np.random.default_rng(seed)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        self.range_width = self.upper_bounds - self.lower_bounds

        # PSO 相关变量
        self.positions = None
        self.velocities = None

        self.pbest_position = None
        self.pbest_values = None

        self.gbest_position = None
        self.gbest_value = None

        self.history = []

        self.trf_result = None
        self.final_position = None
        self.final_cost = None

        # 初始化粒子群
        self._initialize_particles()

    def _loss(self, params):
        """
         把残差向量转换成 PSO 可以优化的标量目标函数。

         least_squares 优化的是：
            0.5 * sum(residuals ** 2)

        这里保持同样的定义。
        """
        try:
            residuals = self.func(params, *self.args)
            residuals = np.asarray(residuals, dtype=float).ravel()

            if not np.all(np.isfinite(residuals)):
                return np.inf

            return 0.5 * np.sum(residuals ** 2)

        except Exception:
            return np.inf

    def _initialize_particles(self):
        #初始化粒子位置
        self.positions = self.rng.uniform(self.lower_bounds, self.upper_bounds, size=(self.num_particles, self.dim))
        #初始化粒子速度
        self.velocities = self.rng.uniform(-0.3 * abs(self.range_width), 0.3 * abs(self.range_width),
                                           size=(self.num_particles, self.dim))
        # 初始化每个粒子的历史最好位置 pbest
        self.pbest_position = self.positions.copy()  # 个体历史最好位置
        self.pbest_values = np.array([self._loss(pos) for pos in self.positions])  # 每个粒子最好位置对应的函数值

        #初始化全局最好位置 gbest
        best_index = self._get_best_index(self.pbest_values)  # 获取全局最好的粒子的索引
        self.gbest_position = self.pbest_position[best_index].copy()
        self.gbest_value = self.pbest_values[best_index]

    def _is_better(self, value1, value2):
        '''
        判断是否变得更好
        '''
        return value1 < value2

    def _get_best_index(self, values):
        '''
        找出最优粒子的索引
        '''
        return np.argmin(values)

    def optimize(self, verbose=True):
        '''
        执行优化
        '''
        # 若未初始化，先初始化粒子群
        if self.positions is None:
            self._initialize_particles()

        for iteration in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iter
            for i in range(self.num_particles):
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                #个体认知：朝自己的历史最优靠近
                cognitive = self.c1 * r1 * (self.pbest_position[i] - self.positions[i])

                #群体认知：朝群体最优靠近
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])

                #更新速度
                self.velocities[i] = (w * self.velocities[i] + cognitive + social)

                #限制最大速度
                if self.vmax is not None:
                    self.velocities[i] = np.clip(self.velocities[i], -self.vmax, self.vmax)  # np.clip(x, lo, hi) 把x里所有元素截断到[lo,hi]，小于lo的变lo，大于hi的变hi

                #更新位置
                self.positions[i] = self.positions[i] + self.velocities[i]

                #防止粒子飞出边界
                self.positions[i] = np.clip(self.positions[i], self.lower_bounds, self.upper_bounds)

                #计算当前的函数值
                current_value = self._loss(self.positions[i])

                #更新粒子的pbest
                if self._is_better(current_value, self.pbest_values[i]):
                    self.pbest_values[i] = current_value
                    self.pbest_position[i] = self.positions[i].copy()

            #更新整个粒子群的gbest
            best_index = self._get_best_index(self.pbest_values)
            best_value = self.pbest_values[best_index]

            if self._is_better(best_value, self.gbest_value):
                self.gbest_value = best_value
                self.gbest_position = self.pbest_position[best_index].copy()

            self.history.append(self.gbest_value)

            if verbose:
                print(
                    f"Iteration {iteration + 1}: "
                    f"gbest_position = {self.gbest_position}, "
                    f"gbest_value = {self.gbest_value}"
                )
        return self.gbest_position, self.gbest_value

    def run_trf(self, x0=None, verbose=False):
        '''
        从PSO得到的结果出发，运行TRF局部优化
        '''
        if x0 is None:
            if self.gbest_position is None:
                raise RuntimeError("没有提供初始x0")
            else:
                x0 = self.gbest_position
        x0 = np.asarray(x0, dtype=float)
        x0 = np.clip(x0, self.lower_bounds, self.upper_bounds)

        default_options = {
            "method": "trf",
            "bounds": (self.lower_bounds, self.upper_bounds),
            "max_nfev": 2000,
            "xtol": 1e-10,
            "ftol": 1e-10,
            "gtol": 1e-10,
        }

        default_options.update(self.trf_options)

        self.trf_result = least_squares(
            fun=self.func,
            x0=x0,
            args=self.args,
            **default_options
        )
        self.final_position = self.trf_result.x
        self.final_cost = self.trf_result.cost

        if verbose:
            print("TRF success:", self.trf_result.success)
            print("TRF message:", self.trf_result.message)
            print("TRF cost:", self.trf_result.cost)

        return self.trf_result

    def fit(self, verbose=False):
        '''
        完成PSO全局搜索与TRF局部优化
        '''
        pso_params, pso_loss = self.optimize(verbose=verbose)
        trf_result = self.run_trf(x0=pso_params, verbose=verbose)

        return {
            "pso_params": pso_params,
            "pso_loss": pso_loss,
            "trf_params": trf_result.x,
            "trf_cost": trf_result.cost,
            "trf_success": trf_result.success,
            "trf_message": trf_result.message,
            "trf_result": trf_result,
        }
