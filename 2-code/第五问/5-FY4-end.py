import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.optimize import fmin_l_bfgs_b
from system_at_t import cover_system
from cover_checker import AdvancedMissileSmokeChecker
delta_t1 = 30  # 第一个烟雾弹投放时间范围
delta_t2 = 30  # 第二个烟雾弹投放时间范围
min_theta = 15  # 角度偏移范围
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAdaptiveDE_LBFGS_Optimizer:
    def __init__(self, pop_size: int = 120, f_min: float = 0.5, f_max: float = 1.0,
                 cr_min: float = 0.2, cr_max: float = 1.0, max_generations: int = 300,
                 restart_threshold: float = 5e-6, max_stagnation: int = 15,
                 max_restarts: int = 40, diversity_threshold: float = 0.15,
                 lbfgs_max_iter: int = 40, lbfgs_tol: float = 1e-7):
        # DE参数
        self.pop_size = pop_size
        self.f_min = f_min        
        self.f_max = f_max        
        self.cr_min = cr_min      
        self.cr_max = cr_max      
        self.max_generations = max_generations
        
        # 重启与自适应策略参数
        self.restart_threshold = restart_threshold
        self.max_stagnation = max_stagnation
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.diversity_threshold = diversity_threshold
        
        # L-BFGS参数
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tol = lbfgs_tol
        
        # 问题相关
        self.n_vars = 8  # 保持8个变量维度
        self.checker = AdvancedMissileSmokeChecker()
        self.population = None
        self.best_history = []
        self.history_window = 6
        
        # 自适应状态变量
        self.current_f = np.random.uniform(f_min, f_max)
        self.current_cr = np.random.uniform(cr_min, cr_max)
        self.success_history = []
        self.success_window = 5

    def _initjude(self, idex_FY: int, index_M_bomb: int) -> float:
        """统一角度计算工具函数"""
        FY = [[17800, 0, 1800], [12000, 1400, 1400], 
              [6000, -3000, 700], [11000, 2000, 1800], [13000, -2000, 1300]]
        M_int = [[20000, 0, 2000], [19000, 600, 2100], [18000, -600, 1900]]
        
        Xfy, Yfy = FY[idex_FY][0], FY[idex_FY][1]
        XG = (Xfy + M_int[index_M_bomb][0]) / 3  # 移除冗余的+0
        YG = (Yfy + M_int[index_M_bomb][1] + 200) / 3
        
        xx, yy = XG - Xfy, YG - Yfy
        l = np.hypot(xx, yy)  # 更高效的距离计算
        cos_theta = xx / l if l != 0 else 1.0
        theta = np.arccos(cos_theta)
        return theta if yy >= 0 else 2 * np.pi - theta

    def initialize_population(self) -> np.ndarray:
        """初始化种群：边界采样+均匀分布"""
        pop = np.zeros((self.pop_size, self.n_vars))
        for i in range(self.n_vars):
            # 20%边界样本，80%均匀样本
            boundary_samples = np.random.choice([0, 1], int(self.pop_size*0.2), p=[0.5, 0.5])
            uniform_samples = np.random.rand(self.pop_size - len(boundary_samples))
            pop[:, i] = np.concatenate([boundary_samples, uniform_samples])
            np.random.shuffle(pop[:, i])
        return pop

    def population_diversity(self) -> float:
        """计算种群多样性：空间分布+适应度分布"""
        if self.population is None:
            return 0.0
        # 空间多样性
        space_div = np.mean(np.std(self.population, axis=0)) / (self.population.max() - self.population.min() + 1e-8)
        # 适应度多样性
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        fit_div = np.std(fitness_values) / (np.max(fitness_values) - np.min(fitness_values) + 1e-8)
        return 0.6 * space_div + 0.4 * fit_div

    def adaptive_params(self, diversity: float, success_rate: float):
        """基于多样性和成功率的参数自适应调整"""
        if diversity < self.diversity_threshold:
            self.current_f = min(self.f_max, self.current_f * 1.2)
            self.current_cr = min(self.cr_max, self.current_cr * 1.2)
        else:
            if success_rate > 0.5:
                self.current_f = max(self.f_min, self.current_f * 0.9)
                self.current_cr = max(self.cr_min, self.current_cr * 0.9)
            else:
                self.current_f = min(self.f_max, self.current_f * 1.1)
                self.current_cr = min(self.cr_max, self.current_cr * 1.1)

    def de_mutation(self, target_idx: int) -> np.ndarray:
        """DE/rand/2变异策略"""
        idxs = [i for i in range(self.pop_size) if i != target_idx]
        r1, r2, r3, r4, r5 = np.random.choice(idxs, 5, replace=False)
        return self.population[r1] + self.current_f * (self.population[r2] - self.population[r3]) + \
               self.current_f * 0.5 * (self.population[r4] - self.population[r5])

    def de_crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """混合交叉策略：指数交叉+二项交叉"""
        trial = target.copy()
        if np.random.random() < 0.5:
            # 指数交叉
            start_idx = np.random.randint(self.n_vars)
            length = 0
            while length < self.n_vars and np.random.random() < self.current_cr:
                length += 1
            for i in range(length):
                pos = (start_idx + i) % self.n_vars
                trial[pos] = mutant[pos]
        else:
            # 二项交叉（至少替换一个维度）
            force_idx = np.random.randint(self.n_vars)
            for i in range(self.n_vars):
                if i == force_idx or np.random.random() < self.current_cr:
                    trial[i] = mutant[i]
        return np.clip(trial, 0, 1)

    def restart_population(self, best_individual: np.ndarray) -> np.ndarray:
        """种群重启策略：保留优质解并引入扰动"""
        self.restart_count += 1
        logger.warning(f"种群重启，第{self.restart_count}次重启（增强多样性）")
        
        new_pop = np.zeros((self.pop_size, self.n_vars))
        sorted_indices = np.argsort([self.fitness(ind) for ind in self.population])[::-1]
        top3 = [self.population[i] for i in sorted_indices[:3]]
        
        new_pop[0] = best_individual.copy()
        new_pop[1] = self._mutate_individual(best_individual, 0.4)
        new_pop[2] = top3[1].copy() if len(top3) > 1 else self._mutate_individual(best_individual, 0.3)
        new_pop[3] = self._mutate_individual(top3[1], 0.3) if len(top3) > 1 else self._mutate_individual(best_individual, 0.2)
        new_pop[4:] = self.initialize_population()[4:]
        return new_pop

    def _mutate_individual(self, ind: np.ndarray, scale: float) -> np.ndarray:
        """个体变异：非对称扰动"""
        perturbation = scale * (np.random.rand(self.n_vars) - 0.3)  # 偏向正向扰动
        return np.clip(ind + perturbation, 0, 1)

    def is_stagnant(self, current_best: float) -> bool:
        """停滞检测：窗口内变化+显著改进判断"""
        if len(self.best_history) < self.history_window:
            return False
        
        recent = self.best_history[-self.history_window:]
        value_change = np.max(recent) - np.min(recent) < self.restart_threshold
        historical_best = max(self.best_history)
        no_significant_improve = all(v < historical_best * 1.01 for v in recent)
        
        return value_change and no_significant_improve

    def fitness(self, x: np.ndarray) -> float:
        """适应度计算：烟雾弹阻断时间"""
        # 基础参数计算
        theta_int = self._initjude(3, 1)  # 统一使用内部方法计算角度
        
        
        # 速度和角度参数
        vi = np.array([0, 0, 0,  105 + x[0] * 70 - 35])  # v = 70~140
        theta = np.array([0, 0, 0,  theta_int + (x[1] * min_theta*2 - min_theta) * np.pi / 180])  # 角度偏移±15度
        
        # 烟雾弹时间参数（仅使用前两个）
        tdrop_1 = x[2] * delta_t1  # 投放时间: 0~20s
        texpl_1 = tdrop_1 + x[3] * 20  # 爆炸时间: 投放后0~20s
        
        tdrop_2 = tdrop_1 + 1 + x[4] * delta_t2  # 投放时间: 第一个+1s后，再0~20s
        texpl_2 = tdrop_2 + x[5] * 20  # 爆炸时间: 投放后0~20s
        
        # 第三个烟雾弹未启用（保持与原逻辑一致）
        tdrop = np.array([[100000, 100000, 100000],
                          [100000, 100000, 100000],
                          [100000, 100000, 100000], 
                          [tdrop_1, tdrop_2, 100000]])
        texpl = np.array([[100000, 100000, 100000],
                          [100000, 100000, 100000],
                          [100000, 100000, 100000],                        
                          [texpl_1, texpl_2, 100000]])

        # 计算阻断时间
        cover = cover_system(vi, theta, tdrop, texpl)
        pace = 100
        t_block = 0
        delta_t = 67 / pace

        for t in np.linspace(0, 67, pace):
            Mj, smokes_location = cover(t, 1)  # 追逐M2
            if len(smokes_location) > 0 and self.checker.check(Mj, smokes_location):
                t_block += delta_t
        
        return t_block

    def _lbfgs_obj(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """L-BFGS目标函数（负适应度）"""
        f = -self.fitness(x)
        epsilon = 1e-5
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = np.clip(x.copy() + np.eye(len(x))[i] * epsilon, 0, 1)
            x_minus = np.clip(x.copy() - np.eye(len(x))[i] * epsilon, 0, 1)
            grad[i] = (-self.fitness(x_plus) + self.fitness(x_minus)) / (2 * epsilon)
        return f, grad

    def lbfgs_optimize(self, x0: np.ndarray) -> np.ndarray:
        """L-BFGS局部优化：多初始点扰动"""
        best_x = x0
        best_f = -np.inf
        
        for i in range(3):
            perturbed_x0 = self._mutate_individual(x0, 0.05 * (i+1))
            x, f, _ = fmin_l_bfgs_b(
                self._lbfgs_obj,
                perturbed_x0,
                bounds=[(0, 1)] * self.n_vars,
                maxiter=self.lbfgs_max_iter,
                factr=self.lbfgs_tol * 1e7,
                pgtol=self.lbfgs_tol,
                disp=0
            )
            current_f = -f
            if current_f > best_f:
                best_f = current_f
                best_x = x
        return best_x

    def optimize(self) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """主优化流程"""
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
        self.best_history.append(global_best_fitness)

        stagnation_count = 0

        for gen in range(self.max_generations):
            diversity = self.population_diversity()
            success_rate = np.mean(self.success_history) if self.success_history else 0.5
            self.adaptive_params(diversity, success_rate)

            # 生成子代
            offspring = []
            current_success = 0
            for i in range(self.pop_size):
                mutant = self.de_mutation(i)
                trial = self.de_crossover(population[i], mutant)
                offspring.append(trial)
                if self.fitness(trial) > fitness_values[i]:
                    current_success += 1

            # 更新成功历史
            self.success_history.append(current_success / self.pop_size)
            if len(self.success_history) > self.success_window:
                self.success_history.pop(0)

            # 评估与选择
            offspring = np.array(offspring)
            offspring_fitness = np.array([self.fitness(ind) for ind in offspring])
            for i in range(self.pop_size):
                if offspring_fitness[i] > fitness_values[i]:
                    population[i] = offspring[i]
                    fitness_values[i] = offspring_fitness[i]

            # 更新全局最优
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_individual = population[current_best_idx].copy()

            if current_best_fitness > global_best_fitness * 1.001:
                global_best_fitness = current_best_fitness
                global_best_individual = current_best_individual.copy()
                stagnation_count = 0
                logger.info(f"第{gen+1}代找到新全局最优: {global_best_fitness:.6f}")
            else:
                stagnation_count += 1

            # 记录与日志
            best_history.append(current_best_fitness)
            avg_history.append(np.mean(fitness_values))
            self.best_history.append(current_best_fitness)
            if len(self.best_history) > self.history_window * 2:
                self.best_history = self.best_history[-self.history_window * 2:]

            logger.info(
                f"第{gen+1}代 - 最优: {current_best_fitness:.6f}, "
                f"平均: {avg_history[-1]:.6f}, 多样性: {diversity:.4f}, "
                f"F: {self.current_f:.3f}, CR: {self.current_cr:.3f}, "
                f"成功率: {current_success/self.pop_size:.2f}, 停滞: {stagnation_count}"
            )

            # 重启判断
            restart_conditions = [
                stagnation_count >= self.max_stagnation,
                self.is_stagnant(current_best_fitness),
                diversity < self.diversity_threshold * 0.5
            ]
            
            if any(restart_conditions) and self.restart_count < self.max_restarts:
                population = self.restart_population(global_best_individual)
                fitness_values = np.array([self.fitness(ind) for ind in population])
                self.population = population
                stagnation_count = 0
                self.current_f = np.random.uniform(self.f_min, self.f_max)
                self.current_cr = np.random.uniform(self.cr_min, self.cr_max)
                logger.info(f"重启后初始最优: {np.max(fitness_values):.6f}")

        # 局部优化
        logger.info("开始L-BFGS局部优化...")
        sorted_indices = np.argsort(fitness_values)[::-1]
        candidates = [global_best_individual]
        step = max(1, len(sorted_indices) // 8)
        candidates.extend([population[i] for i in sorted_indices[::step][:8]])
        candidates.extend([self._mutate_individual(global_best_individual, s) 
                          for s in [0.05, 0.1, 0.15, 0.2]])

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

    def denormalize_params(self, x: np.ndarray) -> dict:
        """参数反标准化：与fitness函数严格对应"""
        theta_int = self._initjude(4, 1)  # 与fitness中使用相同的基准角度
        
        # 烟雾弹时间计算（与fitness保持一致）
        
        tdrop_1 = x[2] * delta_t1
        texpl_1 = tdrop_1 + x[3] * 20
        tdrop_2 = tdrop_1 + 1 + x[4] * delta_t2
        texpl_2 = tdrop_2 + x[5] * 20

        return {
            "速度 (m/s)": 105 + x[0] * 70 - 35,  # 70~140范围
            "角度 (度)": np.rad2deg(theta_int + (x[1] * min_theta*2 - min_theta) * np.pi / 180),  # ±15度偏移
            "第一个烟雾弹投放时间 (s)": tdrop_1,
            "第一个烟雾弹爆炸时间 (s)": texpl_1,
            "第二个烟雾弹投放时间 (s)": tdrop_2,
            "第二个烟雾弹爆炸时间 (s)": texpl_2,
            "备用变量3 (未使用)": x[6],  # 原第三个烟雾弹相关变量
            "备用变量4 (未使用)": x[7]   # 原第三个烟雾弹相关变量
        }


if __name__ == "__main__":
    # 初始化优化器
    optimizer = EnhancedAdaptiveDE_LBFGS_Optimizer(
        pop_size=120,
        max_generations=300,
        max_restarts=20
    )
    best_ind, best_fit, best_history, avg_history = optimizer.optimize()
    
    # 可视化优化过程
    plt.figure(figsize=(12, 6))
    plt.plot(best_history, label='最佳适应度（阻断时间）', color='red', linewidth=2)
    plt.plot(avg_history, label='平均适应度', color='blue', linestyle='--', linewidth=1.5)
    plt.axhline(y=5.0, color='green', linestyle=':', label='目标阈值（5秒）')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值（秒）')
    plt.title('增强型DE+L-BFGS优化过程')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('enhanced_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出结果
    actual_params = optimizer.denormalize_params(best_ind)
    print("\n===== 优化结果 =====")
    print(f"最佳阻断时间: {best_fit:.6f} 秒")
    print("\n===== 实际物理参数 =====")
    for name, value in actual_params.items():
        print(f"{name}: {value:.6f}")
    print("\n===== 标准化参数 (0-1) =====")
    print(f"最佳参数 (8个变量): {best_ind}")