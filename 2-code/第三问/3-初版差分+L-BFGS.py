import numpy as np
import logging
from typing import Tuple, List
from scipy.optimize import fmin_l_bfgs_b
from system_at_t import cover_system
from cover_checker import AdvancedMissileSmokeChecker

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveDE_LBFGS_Optimizer:
    def __init__(self, pop_size: int = 100, f_min: float = 0.4, f_max: float = 0.9,
                 cr_min: float = 0.1, cr_max: float = 0.9, max_generations: int = 150,
                 restart_threshold: float = 1e-6, max_stagnation: int = 10,
                 max_restarts: int = 10, diversity_threshold: float = 0.2,
                 lbfgs_max_iter: int = 30, lbfgs_tol: float = 1e-7):
        # DE参数
        self.pop_size = pop_size  # 种群规模（小于原算法，提高效率）
        self.f_min = f_min        # 变异因子最小值
        self.f_max = f_max        # 变异因子最大值
        self.cr_min = cr_min      # 交叉率最小值
        self.cr_max = cr_max      # 交叉率最大值
        self.max_generations = max_generations  # 最大迭代次数（少于原算法）
        
        # 重启与自适应策略参数
        self.restart_threshold = restart_threshold
        self.max_stagnation = max_stagnation  # 更早检测停滞
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.diversity_threshold = diversity_threshold
        
        # L-BFGS局部优化参数
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tol = lbfgs_tol
        
        # 问题相关
        self.n_vars = 8  # 8个优化变量
        self.checker = AdvancedMissileSmokeChecker()
        self.population = None
        self.best_history = []  # 记录历史最优解用于重启判断
        self.history_window = 8  # 窗口大小小于原算法
        
        # 自适应状态变量
        self.current_f = (f_min + f_max) / 2  # 当前变异因子
        self.current_cr = (cr_min + cr_max) / 2  # 当前交叉率

    def initialize_population(self) -> np.ndarray:
        """拉丁超立方采样初始化种群，保证初始多样性"""
        pop = np.zeros((self.pop_size, self.n_vars))
        for i in range(self.n_vars):
            intervals = np.linspace(0, 1, self.pop_size + 1)
            for j in range(self.pop_size):
                pop[j, i] = np.random.uniform(intervals[j], intervals[j+1])
        return pop

    def population_diversity(self) -> float:
        """计算种群多样性（空间分布为主，降低计算成本）"""
        if self.population is None:
            return 0.0
        # 仅计算空间多样性，减少适应度计算开销
        return np.mean(np.std(self.population, axis=0))

    def adaptive_params(self, diversity: float):
        """根据种群多样性自适应调整DE的F和CR"""
        # 多样性低时：增大F促进探索，提高CR增加信息交换
        if diversity < self.diversity_threshold:
            self.current_f = min(self.f_max, self.current_f * 1.1)
            self.current_cr = min(self.cr_max, self.current_cr * 1.1)
        # 多样性高时：减小F促进收敛，降低CR保持优良基因
        else:
            self.current_f = max(self.f_min, self.current_f * 0.9)
            self.current_cr = max(self.cr_min, self.current_cr * 0.9)

    def de_mutation(self, target_idx: int) -> np.ndarray:
        """DE/rand/1变异策略：v = x_r1 + F*(x_r2 - x_r3)"""
        # 随机选择3个不同的个体（与目标个体不同）
        idxs = [i for i in range(self.pop_size) if i != target_idx]
        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
        return self.population[r1] + self.current_f * (self.population[r2] - self.population[r3])

    def de_crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """二项式交叉，保证至少一个维度被变异"""
        trial = target.copy()
        # 随机选择一个维度强制变异，避免trial与target完全相同
        force_idx = np.random.randint(self.n_vars)
        for i in range(self.n_vars):
            if i == force_idx or np.random.random() < self.current_cr:
                trial[i] = mutant[i]
        # 边界处理
        return np.clip(trial, 0, 1)

    def restart_population(self, best_individual: np.ndarray) -> np.ndarray:
        """重启策略：保留最优个体并引入多样化新个体"""
        self.restart_count += 1
        logger.info(f"种群重启，第{self.restart_count}次重启")
        
        # 保留最优个体及其变异体（增加局部多样性）
        new_pop = np.zeros((self.pop_size, self.n_vars))
        new_pop[0] = best_individual.copy()
        new_pop[1] = self._mutate_individual(best_individual, 0.3)  # 强变异
        
        # 剩余个体用新采样填充
        new_pop[2:] = self.initialize_population()[2:]
        return new_pop

    def _mutate_individual(self, ind: np.ndarray, scale: float) -> np.ndarray:
        """个体变异辅助函数，用于重启时增加多样性"""
        mutated = ind + scale * (np.random.rand(self.n_vars) - 0.5)
        return np.clip(mutated, 0, 1)

    def is_stagnant(self, current_best: float) -> bool:
        """判断是否停滞（最近窗口内最优解无显著改进）"""
        if len(self.best_history) < self.history_window:
            return False
        # 检查最近窗口的最优值变化
        recent = self.best_history[-self.history_window:]
        return np.max(recent) - np.min(recent) < self.restart_threshold

    def fitness(self, x: np.ndarray) -> float:
        """原封不动保留适应度计算逻辑（问题核心）"""
        def initjude(idex_FY, index_M_bomb):
            FY = [[17800, 0, 1800], [12000, 1400, 1400], [6000, -3000, 700], [11000, 2000, 1800], [13000, -2000, 1300]]
            M_int = [[20000, 0, 2000], [19000, 600, 2100], [18000, -600, 1900]]
            i = idex_FY
            j = index_M_bomb
            Xfy = FY[i][0]
            Yfy = FY[i][1]
            XG = (Xfy + M_int[j][0] + 0) / 3
            YG = (Yfy + M_int[j][1] + 200) / 3
            xx = XG - Xfy
            yy = YG - Yfy
            l = (xx**2 + yy**2)** 0.5
            cos = xx / ((xx**2 + yy**2)** 0.5) if l != 0 else 1
            theta = np.arccos(cos)
            if yy < 0:
                theta = 2 * np.pi - theta
            return theta

        theta_int_11 = initjude(0, 0)
        vi = 105 + np.array([x[0] * 70 - 35])
        theta = theta_int_11 + np.array([(x[1] * 30 - 15) * np.pi / 180])
        
        delta_t1 = 15
        delta_t2 = 30
        delta_t3 = 30
        
        tdrop_11 = x[2] * delta_t1
        texpl_11 = tdrop_11 + x[3] * 20
        
        tdrop_12 = tdrop_11 + 1 + x[4] * delta_t2
        texpl_12 = tdrop_12 + x[5] * 20
        
        tdrop_13 = tdrop_12 + 1 + x[6] * delta_t3
        texpl_13 = tdrop_13 + x[7] * 20
        
        tdrop = np.array([[tdrop_11, tdrop_12, tdrop_13]])
        texpl = np.array([[texpl_11, texpl_12, texpl_13]])

        cover = cover_system(vi, theta, tdrop, texpl)
        pace = 300
        t_block = 0
        delta_t = 67 / pace

        for t in np.linspace(0, 67, pace):
            Mj, smokes_location = cover(t, 1)
            if len(smokes_location) > 0 and self.checker.check(Mj, smokes_location):
                t_block += delta_t

        return t_block

    def _lbfgs_obj(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """L-BFGS目标函数（负适应度+梯度）"""
        f = -self.fitness(x)
        # 数值梯度（与原牛顿法一致，但仅用于L-BFGS）
        epsilon = 1e-5
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_plus = np.clip(x_plus, 0, 1)
            
            x_minus = x.copy()
            x_minus[i] -= epsilon
            x_minus = np.clip(x_minus, 0, 1)
            
            grad[i] = (-self.fitness(x_plus) + self.fitness(x_minus)) / (2 * epsilon)
        return f, grad

    def lbfgs_optimize(self, x0: np.ndarray) -> np.ndarray:
        """L-BFGS局部优化（替代牛顿法）"""
        # 带边界约束的L-BFGS优化
        x, f, d = fmin_l_bfgs_b(
            self._lbfgs_obj,
            x0,
            bounds=[(0, 1)] * self.n_vars,
            maxiter=self.lbfgs_max_iter,
            factr=self.lbfgs_tol * 1e7,  # 转换为L-BFGS的factr参数
            pgtol=self.lbfgs_tol,
            disp=0
        )
        logger.debug(f"L-BFGS优化完成，迭代次数: {d['nit']}, 函数评估: {d['funcalls']}")
        return x

    def optimize(self) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """主优化流程：DE全局搜索 + L-BFGS局部精修"""
        # 初始化种群
        population = self.initialize_population()
        fitness_values = np.array([self.fitness(ind) for ind in population])
        self.population = population

        # 初始化全局最优
        global_best_idx = np.argmax(fitness_values)
        global_best_fitness = fitness_values[global_best_idx]
        global_best_individual = population[global_best_idx].copy()

        # 记录历史
        best_history = [global_best_fitness]
        avg_history = [np.mean(fitness_values)]
        self.best_history.append(global_best_individual)

        stagnation_count = 0

        for gen in range(self.max_generations):
            # 计算种群多样性
            diversity = self.population_diversity()
            # 自适应调整DE参数
            self.adaptive_params(diversity)

            # 生成子代种群
            offspring = []
            for i in range(self.pop_size):
                # 变异
                mutant = self.de_mutation(i)
                # 交叉
                trial = self.de_crossover(population[i], mutant)
                offspring.append(trial)
            offspring = np.array(offspring)

            # 评估子代
            offspring_fitness = np.array([self.fitness(ind) for ind in offspring])

            # 选择（贪婪选择）
            for i in range(self.pop_size):
                if offspring_fitness[i] > fitness_values[i]:
                    population[i] = offspring[i]
                    fitness_values[i] = offspring_fitness[i]

            # 更新全局最优
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_individual = population[current_best_idx].copy()

            if current_best_fitness > global_best_fitness:
                global_best_fitness = current_best_fitness
                global_best_individual = current_best_individual.copy()
                stagnation_count = 0  # 重置停滞计数
                logger.info(f"第{gen+1}代找到新全局最优: {global_best_fitness:.6f}")
            else:
                stagnation_count += 1

            # 记录历史
            best_history.append(current_best_fitness)
            avg_history.append(np.mean(fitness_values))
            self.best_history.append(current_best_individual)
            if len(self.best_history) > self.history_window * 2:
                self.best_history = self.best_history[-self.history_window * 2:]

            # 打印状态
            logger.info(
                f"第{gen+1}代 - 最优: {current_best_fitness:.6f}, "
                f"平均: {avg_history[-1]:.6f}, 多样性: {diversity:.4f}, "
                f"F: {self.current_f:.3f}, CR: {self.current_cr:.3f}, 停滞: {stagnation_count}"
            )

            # 检查重启条件
            if (stagnation_count >= self.max_stagnation or 
                self.is_stagnant(current_best_fitness)) and self.restart_count < self.max_restarts:
                population = self.restart_population(global_best_individual)
                fitness_values = np.array([self.fitness(ind) for ind in population])
                self.population = population
                stagnation_count = 0
                logger.info(f"重启后初始最优: {np.max(fitness_values):.6f}")

        # L-BFGS局部精修（多起点）
        logger.info("开始L-BFGS局部优化...")
        # 选择候选解（全局最优+Top5个体+变异体）
        sorted_indices = np.argsort(fitness_values)[::-1]
        candidates = [global_best_individual]
        candidates.extend(population[sorted_indices[:5]])
        # 加入变异的候选解扩大搜索
        candidates.extend([self._mutate_individual(global_best_individual, 0.1) for _ in range(2)])

        # 局部优化并选择最佳
        best_refined_fitness = -np.inf
        best_refined_individual = None
        for i, cand in enumerate(candidates):
            logger.info(f"优化第{i+1}/{len(candidates)}个候选解")
            refined = self.lbfgs_optimize(cand)
            refined_fitness = self.fitness(refined)
            if refined_fitness > best_refined_fitness:
                best_refined_fitness = refined_fitness
                best_refined_individual = refined

        # 确定最终结果
        if best_refined_fitness > global_best_fitness:
            logger.info(f"局部优化提升: {global_best_fitness:.6f} → {best_refined_fitness:.6f}")
            final_ind = best_refined_individual
            final_fit = best_refined_fitness
            best_history.append(final_fit)
            avg_history.append(avg_history[-1])
        else:
            logger.info("局部优化未提升，保留全局最优")
            final_ind = global_best_individual
            final_fit = global_best_fitness

        return final_ind, final_fit, best_history, avg_history


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 初始化优化器（参数经过调优）
    optimizer = AdaptiveDE_LBFGS_Optimizer(
        pop_size=100,
        max_generations=150,
        max_restarts=3
    )
    best_ind, best_fit, best_history, avg_history = optimizer.optimize()
    
    # 可视化优化过程
    plt.figure(figsize=(12, 6))
    plt.plot(best_history, label='最佳适应度（阻断时间）', color='red', linewidth=2)
    plt.plot(avg_history, label='平均适应度', color='blue', linestyle='--', linewidth=1.5)
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值（秒）')
    plt.title('DE+L-BFGS优化过程')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('de_lbfgs_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 参数反标准化（与原代码一致）
    def denormalize_params(x):
        def initjude(idex_FY, index_M_bomb):
            FY = [[17800, 0, 1800], [12000, 1400, 1400], [6000, -3000, 700], [11000, 2000, 1800], [13000, -2000, 1300]]
            M_int = [[20000, 0, 2000], [19000, 600, 2100], [18000, -600, 1900]]
            i, j = idex_FY, index_M_bomb
            Xfy, Yfy = FY[i][0], FY[i][1]
            XG = (Xfy + M_int[j][0] + 0) / 3
            YG = (Yfy + M_int[j][1] + 200) / 3
            xx, yy = XG - Xfy, YG - Yfy
            l = (xx**2 + yy**2)** 0.5
            cos = xx / l if l != 0 else 1
            theta = np.arccos(cos)
            return theta if yy >= 0 else 2 * np.pi - theta
        
        theta_int_11 = initjude(0, 0)
        
        params = {
            "速度 (m/s)": 105 + x[0] * 70 - 35,
            "角度 (度)": np.rad2deg(theta_int_11 + (x[1] * 30 - 15) * np.pi / 180),
            "第一个烟雾弹投放时间 (s)": x[2] * 15,
            "第一个烟雾弹爆炸时间 (s)": x[2] * 15 + x[3] * 20,
            "第二个烟雾弹投放时间 (s)": x[2] * 15 + 1 + x[4] * 30,
            "第二个烟雾弹爆炸时间 (s)": (x[2] * 15 + 1 + x[4] * 30) + x[5] * 20,
            "第三个烟雾弹投放时间 (s)": (x[2] * 15 + 1 + x[4] * 30) + 1 + x[6] * 30,
            "第三个烟雾弹爆炸时间 (s)": ((x[2] * 15 + 1 + x[4] * 30) + 1 + x[6] * 30) + x[7] * 20
        }
        return params
    
    # 输出结果
    actual_params = denormalize_params(best_ind)
    print("\n===== 优化结果 =====")
    print(f"最佳阻断时间: {best_fit:.6f} 秒")
    print("\n===== 实际物理参数 =====")
    for name, value in actual_params.items():
        print(f"{name}: {value:.6f}")
    print("\n===== 标准化参数 (0-1) =====")
    print(f"最佳参数 (8个变量): {best_ind}")