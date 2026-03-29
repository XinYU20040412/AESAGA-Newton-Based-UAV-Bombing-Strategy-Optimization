import numpy as np
import logging
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
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
# 假设这些是原代码中定义的常量（根据5_FY3.py推断）
theta_int = initjude(2, 2)  # 角度基准值，实际值可能需要根据原代码调整
min_theta = 30  # 角度最小值，实际值可能需要根据原代码调整
delta_trop = 60  # 投放时间范围，实际值可能需要根据原代码调整
delta_t_expl_diff = 20  # 爆炸时间差范围，实际值可能需要根据原代码调整

# 假设的烟雾系统和检查器类（根据5_FY3.py推断）
class cover_system:
    """烟雾系统类，用于计算特定时间的烟雾位置"""
    def __init__(self, vi, theta, tdrop, texpl):
        self.vi = vi
        self.theta = theta
        self.tdrop = tdrop
        self.texpl = texpl
    
    def __call__(self, t, mode):
        # 这里是简化实现，实际应根据原代码逻辑实现
        Mj = np.array([10000 + t * 10, 0, 1500])  # 示例目标位置
        smokes_location = []
        # 检查烟雾是否已投放且已爆炸
        for i in range(len(self.tdrop)):
            for j in range(len(self.tdrop[i])):
                if self.tdrop[i][j] <= t <= self.texpl[i][j]:
                    # 计算烟雾位置（简化模型）
                    x = self.vi[i] * np.cos(self.theta[i]) * (t - self.tdrop[i][j])
                    y = self.vi[i] * np.sin(self.theta[i]) * (t - self.tdrop[i][j])
                    z = 1500  # 固定高度示例
                    smokes_location.append(np.array([x, y, z]))
        return Mj, smokes_location

class AdvancedMissileSmokeChecker:
    """烟雾检查器类，用于检查烟雾是否阻断目标"""
    def check(self, Mj, smokes_location):
        # 简化实现：检查目标是否在任何烟雾范围内
        for smoke in smokes_location:
            distance = np.linalg.norm(Mj - smoke)
            if distance < 500:  # 假设500单位内视为被阻断
                return True
        return False

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
        
        # 问题相关（使用5_FY3.py中的4个变量）
        self.n_vars = 4  # 5_FY3.py中使用4个变量
        self.checker = AdvancedMissileSmokeChecker()
        self.population = None
        self.best_history = []
        self.history_window = 6
        
        # 自适应状态变量
        self.current_f = np.random.uniform(f_min, f_max)
        self.current_cr = np.random.uniform(cr_min, cr_max)
        self.success_history = []
        self.success_window = 5

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
        """计算个体适应度（阻断时间）- 来自5_FY3.py"""
        # 将比例参数转换为实际参数
        vi = np.array([0, 0, x[0] * 70 + 70])  # 速度参数转换
        theta = np.array([0, 0, theta_int + ((x[1] * 2 * min_theta - min_theta) * np.pi / 180)])  # 角度参数转换
        
        # 时间参数转换
        tdrop = np.array([
            [100000, 100000, 100000],
            [100000, 100000, 100000],
            [x[2] * delta_trop, 100000, 100000]
        ])
        texpl = np.array([
            [100000, 100000, 100000],
            [100000, 100000, 100000],
            [x[3] * delta_t_expl_diff + x[2] * delta_trop, 100000, 100000]
        ])

        # 创建烟雾系统实例
        cover = cover_system(vi, theta, tdrop, texpl)

        # 计算阻断时间 - 增加精度
        pace = 200  # 提高采样精度
        t_block = 0
        delta_t = 67 / pace

        for t in np.linspace(0, 67, pace):
            Mj, smokes_location = cover(t, 2)
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
        """主优化流程 - 来自5-FY4.py"""
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

    def denormalize_params(self, x: np.ndarray) -> Dict[str, float]:
        """参数反标准化：与fitness函数严格对应"""
        # 与5_FY3.py中的fitness函数保持一致的反标准化
        vi = x[0] * 70 + 70  # 速度
        theta = theta_int + ((x[1] * 2 * min_theta - min_theta) * np.pi / 180)  # 角度（弧度）
        tdrop = x[2] * delta_trop  # 投放时间
        texpl = x[3] * delta_t_expl_diff + x[2] * delta_trop  # 爆炸时间
        
        return {
            "速度 (m/s)": vi,
            "角度 (度)": np.rad2deg(theta),
            "烟雾弹投放时间 (s)": tdrop,
            "烟雾弹爆炸时间 (s)": texpl
        }

def visualize_results(best_history: List[float], avg_history: List[float], 
                     params: Dict[str, float], final_fitness: float):
    """可视化优化结果"""
    # 绘制适应度历史曲线
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(best_history, label='FY3最优适应度', color='blue')
    ax1.plot(avg_history, label='平均适应度', color='orange', alpha=0.7)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('FY3适应度（阻断时间/s）')
    ax1.set_title('优化过程中的适应度变化')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('适应度变化曲线.png', dpi=300)
    plt.show()
    
    # 绘制最终参数条形图
    fig, ax2 = plt.subplots(figsize=(10, 6))
    params_names = list(params.keys())
    params_values = list(params.values())
    
    ax2.bar(params_names, params_values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
    ax2.set_title(f'最优参数（阻断时间: {final_fitness:.2f}s）')
    ax2.set_ylabel('参数值')
    plt.xticks(rotation=45, ha='right')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(params_values):
        ax2.text(i, v + max(params_values)*0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('最优参数条形图.png', dpi=300)
    plt.show()

def main():
    """主函数：运行优化并可视化结果"""
    logger.info("开始烟雾弹阻断时间优化...")
    
    # 创建优化器实例
    optimizer = EnhancedAdaptiveDE_LBFGS_Optimizer(
        pop_size=100,
        max_generations=200,
        max_stagnation=20,
        max_restarts=10
    )
    
    # 执行优化
    best_ind, best_fit, best_history, avg_history = optimizer.optimize()
    
    # 反标准化参数
    best_params = optimizer.denormalize_params(best_ind)
    
    # 输出结果
    logger.info("\n优化完成！")
    logger.info(f"最优阻断时间: {best_fit:.6f}秒")
    logger.info("最优参数:")
    for name, value in best_params.items():
        logger.info(f"  {name}: {value:.6f}")
    
    # 可视化结果
    visualize_results(best_history, avg_history, best_params, best_fit)

if __name__ == "__main__":
    main()