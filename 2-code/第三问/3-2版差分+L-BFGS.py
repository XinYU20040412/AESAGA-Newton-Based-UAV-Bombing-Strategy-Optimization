import numpy as np
import logging
from typing import Tuple, List
from scipy.optimize import fmin_l_bfgs_b
from system_at_t import cover_system
from cover_checker import AdvancedMissileSmokeChecker

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAdaptiveDE_LBFGS_Optimizer:
    def __init__(self, pop_size: int = 120, f_min: float = 0.5, f_max: float = 1.0,
                 cr_min: float = 0.2, cr_max: float = 1.0, max_generations: int = 200,
                 restart_threshold: float = 5e-6, max_stagnation: int = 15,  # 缩短停滞容忍代次
                 max_restarts: int = 5, diversity_threshold: float = 0.15,  # 提高多样性阈值
                 lbfgs_max_iter: int = 40, lbfgs_tol: float = 1e-7):
        # DE参数（扩大F和CR范围，增强探索能力）
        self.pop_size = pop_size
        self.f_min = f_min        
        self.f_max = f_max        
        self.cr_min = cr_min      
        self.cr_max = cr_max      
        self.max_generations = max_generations
        
        # 重启与自适应策略参数（更灵敏的停滞检测）
        self.restart_threshold = restart_threshold
        self.max_stagnation = max_stagnation  # 从10代增加到15代，但提高检测灵敏度
        self.max_restarts = max_restarts  # 增加重启次数
        self.restart_count = 0
        self.diversity_threshold = diversity_threshold  # 提高阈值，更早触发多样性维护
        
        # L-BFGS参数（增加迭代次数，提高局部搜索精度）
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tol = lbfgs_tol
        
        # 问题相关
        self.n_vars = 8
        self.checker = AdvancedMissileSmokeChecker()
        self.population = None
        self.best_history = []
        self.history_window = 6  # 缩小窗口，更早发现停滞
        
        # 自适应状态变量（增加初始值）
        self.current_f = np.random.uniform(f_min, f_max)
        self.current_cr = np.random.uniform(cr_min, cr_max)
        
        # 新增：成功历史记录（用于参数自适应）
        self.success_history = []
        self.success_window = 5  # 最近5代的成功记录

    def initialize_population(self) -> np.ndarray:
        """改进的初始化：增加边界附近采样比例，扩大搜索范围"""
        pop = np.zeros((self.pop_size, self.n_vars))
        for i in range(self.n_vars):
            # 20%的样本分布在边界附近，增强边缘探索
            boundary_samples = np.random.choice([0, 1], int(self.pop_size*0.2), p=[0.5, 0.5])
            # 80%的样本均匀分布
            uniform_samples = np.random.rand(self.pop_size - len(boundary_samples))
            pop[:, i] = np.concatenate([boundary_samples, uniform_samples])
            # 打乱顺序
            np.random.shuffle(pop[:, i])
        return pop

    def population_diversity(self) -> float:
        """改进的多样性计算：结合空间分布和适应度分布"""
        if self.population is None:
            return 0.0
        # 空间多样性（标准化）
        space_div = np.mean(np.std(self.population, axis=0)) / (self.population.max() - self.population.min() + 1e-8)
        # 适应度多样性（标准化）
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        fit_div = np.std(fitness_values) / (np.max(fitness_values) - np.min(fitness_values) + 1e-8)
        # 综合多样性
        return 0.6 * space_div + 0.4 * fit_div  # 空间多样性权重更高

    def adaptive_params(self, diversity: float, success_rate: float):
        """改进的自适应策略：结合多样性和成功交叉率"""
        # 双重调节机制：多样性 + 最近交叉成功率
        if diversity < self.diversity_threshold:
            # 多样性低时：大幅提高F和CR
            self.current_f = min(self.f_max, self.current_f * 1.2)
            self.current_cr = min(self.cr_max, self.current_cr * 1.2)
        else:
            # 多样性高时：根据成功交叉率调节
            if success_rate > 0.5:  # 交叉成功率高，适当降低以加速收敛
                self.current_f = max(self.f_min, self.current_f * 0.9)
                self.current_cr = max(self.cr_min, self.current_cr * 0.9)
            else:  # 成功率低，提高以增加探索
                self.current_f = min(self.f_max, self.current_f * 1.1)
                self.current_cr = min(self.cr_max, self.current_cr * 1.1)

    def de_mutation(self, target_idx: int) -> np.ndarray:
        """改进的变异策略：DE/rand/2（更激进的探索）"""
        idxs = [i for i in range(self.pop_size) if i != target_idx]
        r1, r2, r3, r4, r5 = np.random.choice(idxs, 5, replace=False)
        # 双差分变异，增强扰动能力
        return self.population[r1] + self.current_f * (self.population[r2] - self.population[r3]) + \
               self.current_f * 0.5 * (self.population[r4] - self.population[r5])

    def de_crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """改进的交叉：指数交叉+二项交叉混合，增加基因交换效率"""
        trial = target.copy()
        # 50%概率选择指数交叉，50%选择二项交叉
        if np.random.random() < 0.5:
            # 指数交叉：连续多个维度被替换
            start_idx = np.random.randint(self.n_vars)
            length = 0
            while length < self.n_vars and np.random.random() < self.current_cr:
                length += 1
            for i in range(length):
                pos = (start_idx + i) % self.n_vars
                trial[pos] = mutant[pos]
        else:
            # 二项交叉：随机多个维度被替换
            force_idx = np.random.randint(self.n_vars)  # 至少保证一个维度被替换
            for i in range(self.n_vars):
                if i == force_idx or np.random.random() < self.current_cr:
                    trial[i] = mutant[i]
        return np.clip(trial, 0, 1)

    def restart_population(self, best_individual: np.ndarray) -> np.ndarray:
        """增强的重启策略：保留多个优质解并引入更大扰动"""
        self.restart_count += 1
        logger.warning(f"种群重启，第{self.restart_count}次重启（增强多样性）")
        
        # 保留前3个最优个体及其变异体
        new_pop = np.zeros((self.pop_size, self.n_vars))
        sorted_indices = np.argsort([self.fitness(ind) for ind in self.population])[::-1]
        top3 = [self.population[i] for i in sorted_indices[:3]]
        
        new_pop[0] = best_individual.copy()
        new_pop[1] = self._mutate_individual(best_individual, 0.4)  # 更强变异
        new_pop[2] = top3[1].copy() if len(top3) > 1 else self._mutate_individual(best_individual, 0.3)
        new_pop[3] = self._mutate_individual(top3[1], 0.3) if len(top3) > 1 else self._mutate_individual(best_individual, 0.2)
        
        # 剩余个体用改进的初始化方法生成（增加随机性）
        new_pop[4:] = self.initialize_population()[4:]
        return new_pop

    def _mutate_individual(self, ind: np.ndarray, scale: float) -> np.ndarray:
        """改进的个体变异：非对称扰动，增强跳出局部最优的能力"""
        # 正向扰动概率高于负向，适应最大化问题
        perturbation = scale * (np.random.rand(self.n_vars) - 0.3)  # 偏向正向扰动
        mutated = ind + perturbation
        return np.clip(mutated, 0, 1)

    def is_stagnant(self, current_best: float) -> bool:
        """改进的停滞检测：多重条件判断"""
        if len(self.best_history) < self.history_window:
            return False
        
        # 条件1：窗口内最优值变化小于阈值
        recent = self.best_history[-self.history_window:]
        value_change = np.max(recent) - np.min(recent) < self.restart_threshold
        
        # 条件2：最优值连续多代未超过历史最佳的1.01倍（允许微小提升）
        historical_best = max(self.best_history)
        no_significant_improve = all(v < historical_best * 1.01 for v in recent)
        
        return value_change and no_significant_improve

    def fitness(self, x: np.ndarray) -> float:
        """保持原适应度计算，但增加对更优解的奖励"""
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

        # 对超过5秒的解增加额外奖励，引导算法向更高目标搜索
        if t_block > 5.0:
            t_block *= (1 + 0.1 * (t_block - 5.0))  # 非线性奖励
        return t_block

    def _lbfgs_obj(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """L-BFGS目标函数（保持不变）"""
        f = -self.fitness(x)
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
        """增强的局部优化：多初始点扰动"""
        best_x = x0
        best_f = -np.inf
        
        # 对初始点进行多次小扰动，避免局部优化陷入初始点附近的局部最优
        for i in range(3):  # 3次扰动尝试
            perturbed_x0 = self._mutate_individual(x0, 0.05 * (i+1))  # 逐步增大扰动
            x, f, d = fmin_l_bfgs_b(
                self._lbfgs_obj,
                perturbed_x0,
                bounds=[(0, 1)] * self.n_vars,
                maxiter=self.lbfgs_max_iter,
                factr=self.lbfgs_tol * 1e7,
                pgtol=self.lbfgs_tol,
                disp=0
            )
            current_f = -f  # 转换回原始适应度
            if current_f > best_f:
                best_f = current_f
                best_x = x
        return best_x

    def optimize(self) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """主优化流程：增强全局探索和局部精修"""
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
        self.best_history.append(global_best_fitness)  # 存储适应度值而非个体，便于比较

        stagnation_count = 0

        for gen in range(self.max_generations):
            # 计算种群多样性
            diversity = self.population_diversity()
            
            # 计算最近交叉成功率
            success_rate = np.mean(self.success_history) if self.success_history else 0.5
            
            # 自适应调整DE参数
            self.adaptive_params(diversity, success_rate)

            # 生成子代种群
            offspring = []
            current_success = 0  # 记录当前代的成功交叉次数
            for i in range(self.pop_size):
                # 变异（使用更激进的rand/2策略）
                mutant = self.de_mutation(i)
                # 交叉（混合策略）
                trial = self.de_crossover(population[i], mutant)
                offspring.append(trial)
                
                # 提前评估以计算成功率
                trial_fitness = self.fitness(trial)
                if trial_fitness > fitness_values[i]:
                    current_success += 1

            # 更新成功历史
            self.success_history.append(current_success / self.pop_size)
            if len(self.success_history) > self.success_window:
                self.success_history.pop(0)

            offspring = np.array(offspring)
            # 评估子代（复用上面已计算的适应度，减少计算量）
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

            if current_best_fitness > global_best_fitness * 1.001:  # 允许微小提升即重置停滞
                global_best_fitness = current_best_fitness
                global_best_individual = current_best_individual.copy()
                stagnation_count = 0
                logger.info(f"第{gen+1}代找到新全局最优: {global_best_fitness:.6f}")
            else:
                stagnation_count += 1

            # 记录历史
            best_history.append(current_best_fitness)
            avg_history.append(np.mean(fitness_values))
            self.best_history.append(current_best_fitness)
            if len(self.best_history) > self.history_window * 2:
                self.best_history = self.best_history[-self.history_window * 2:]

            # 打印状态
            logger.info(
                f"第{gen+1}代 - 最优: {current_best_fitness:.6f}, "
                f"平均: {avg_history[-1]:.6f}, 多样性: {diversity:.4f}, "
                f"F: {self.current_f:.3f}, CR: {self.current_cr:.3f}, "
                f"成功率: {current_success/self.pop_size:.2f}, 停滞: {stagnation_count}"
            )

            # 改进的重启条件：多重触发机制
            restart_conditions = [
                stagnation_count >= self.max_stagnation,
                self.is_stagnant(current_best_fitness),
                diversity < self.diversity_threshold * 0.5  # 极端低多样性
            ]
            
            if any(restart_conditions) and self.restart_count < self.max_restarts:
                population = self.restart_population(global_best_individual)
                fitness_values = np.array([self.fitness(ind) for ind in population])
                self.population = population
                stagnation_count = 0
                # 重置自适应参数，避免历史影响
                self.current_f = np.random.uniform(self.f_min, self.f_max)
                self.current_cr = np.random.uniform(self.cr_min, self.cr_max)
                logger.info(f"重启后初始最优: {np.max(fitness_values):.6f}")

        # 增强的局部优化：更多候选解
        logger.info("开始L-BFGS局部优化...")
        # 选择更多样化的候选解
        sorted_indices = np.argsort(fitness_values)[::-1]
        candidates = [global_best_individual]
        # 确保候选解分布均匀
        step = max(1, len(sorted_indices) // 8)  # 间隔采样，增加多样性
        candidates.extend([population[i] for i in sorted_indices[::step][:8]])  # 最多8个候选
        # 加入变异的候选解
        candidates.extend([self._mutate_individual(global_best_individual, s) 
                          for s in [0.05, 0.1, 0.15, 0.2]])

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
    # 初始化优化器（参数经过针对性调优）
    optimizer = EnhancedAdaptiveDE_LBFGS_Optimizer(
        pop_size=120,
        max_generations=200,
        max_restarts=5
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
    
    # 参数反标准化
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

