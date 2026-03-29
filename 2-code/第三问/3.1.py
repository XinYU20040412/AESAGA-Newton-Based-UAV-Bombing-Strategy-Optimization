import numpy as np
import logging
from typing import Tuple, List
from system_at_t import cover_system
from cover_checker import AdvancedMissileSmokeChecker

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSmokeOptimizer:
    def __init__(self, pop_size: int = 150, elite_size: int = 15, 
                 alpha: float = 0.9, beta: float = 1.0, T_final: float = 1e-8,
                 restart_threshold: float = 1e-6, max_stagnation: int = 15,
                 max_restarts: int = 5, diversity_threshold: float = 0.3,
                 newton_max_iter: int = 50, newton_tol: float = 1e-6):
        # 增加种群规模以提高多样性
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.alpha = alpha  # 温度衰减系数，降低以增加搜索时间
        self.beta = beta    # 自适应调整因子
        self.T_final = T_final  # 终止温度
        self.T0 = None      # 初始温度，将在优化开始时计算
        self.restart_threshold = restart_threshold  # 停滞判断阈值
        self.max_stagnation = max_stagnation        # 最大停滞代数（减少以更早重启）
        self.restart_count = 0                      # 重启计数
        self.max_restarts = max_restarts            # 增加最大重启次数
        self.diversity_threshold = diversity_threshold  # 多样性阈值（降低以更敏感地检测多样性不足）
        self.population = None  # 当前种群
        self.checker = AdvancedMissileSmokeChecker()  # 烟雾遮挡检查器
        
        # 牛顿法参数
        self.newton_max_iter = newton_max_iter
        self.newton_tol = newton_tol
        
        # 新添加的参数：增强全局搜索
        self.mutation_scale = 0.1  # 变异步长
        self.crossover_rate = 0.9  # 基础交叉率
        self.mutation_rate = 0.2   # 基础变异率（提高以增强探索）
        self.immigration_rate = 0.1  # 移民率：定期引入新个体
        
        # 记录历史最优解以防止循环
        self.history_best = []
        self.history_window = 10  # 检查最近10代的最优解

    def initialize_population(self) -> np.ndarray:
        """初始化种群，8个优化变量，使用拉丁超立方采样提高初始多样性"""
        # 替换简单随机采样为拉丁超立方采样，提高初始种群多样性
        n_vars = 8
        pop = np.zeros((self.pop_size, n_vars))
        
        for i in range(n_vars):
            # 拉丁超立方采样
            intervals = np.linspace(0, 1, self.pop_size + 1)
            for j in range(self.pop_size):
                pop[j, i] = np.random.uniform(intervals[j], intervals[j+1])
        
        return pop

    def population_diversity(self) -> float:
        """计算种群多样性，考虑空间分布和适应度分布"""
        if self.population is None or len(self.population) == 0:
            return 0.0
        
        # 空间多样性
        spatial_div = np.mean(np.std(self.population, axis=0))
        
        # 适应度多样性
        fitness_values = np.array([self.fitness(ind) for ind in self.population])
        fitness_div = np.std(fitness_values) / (np.max(fitness_values) - np.min(fitness_values) + 1e-10)
        
        # 综合多样性指标
        return 0.7 * spatial_div + 0.3 * fitness_div

    def selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """改进的选择操作：锦标赛选择 + 精英保留，平衡探索与利用"""
        selected = []
        
        # 锦标赛选择
        tournament_size = 3
        for _ in range(self.pop_size - self.elite_size):
            # 随机选择锦标赛参与者
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness_values[indices]
            
            # 计算选择概率，修复可能出现的NaN问题
            sum_fitness = np.sum(tournament_fitness)
            if sum_fitness == 0 or np.isnan(sum_fitness):
                # 如果所有适应度都为0或出现NaN，使用均匀概率
                probs = np.ones_like(tournament_fitness) / len(tournament_fitness)
            else:
                # 按概率选择获胜者，适应度高的概率更大
                probs = tournament_fitness / sum_fitness
                # 确保没有NaN值
                probs = np.nan_to_num(probs)
                # 确保概率和为1
                probs = probs / np.sum(probs)
            
            winner_idx = np.random.choice(indices, p=probs)
            selected.append(winner_idx)
            
        return np.array(selected)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """改进的交叉操作：模拟二进制交叉(SBX)，适合浮点数编码"""
        child1, child2 = parent1.copy(), parent2.copy()
        n = len(parent1)
        eta = 20  # 分布指数，控制交叉强度
        
        for i in range(n):
            if np.random.random() < self.crossover_rate:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    # 确保父代有差异时才进行有效交叉
                    if parent1[i] > parent2[i]:
                        # 交换父代值，确保parent1[i] <= parent2[i]
                        parent1[i], parent2[i] = parent2[i], parent1[i]
                    
                    # SBX交叉公式
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        beta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    # 修复：移除方括号，确保是数值计算而非列表
                    child1[i] = 0.5 * ((parent1[i] + parent2[i]) - beta * (parent2[i] - parent1[i]))
                    child2[i] = 0.5 * ((parent1[i] + parent2[i]) + beta * (parent2[i] - parent1[i]))
                    
                    # 边界处理
                    child1[i] = np.clip(child1[i], 0, 1)
                    child2[i] = np.clip(child2[i], 0, 1)
        
        return child1, child2

    def mutation(self, individual: np.ndarray, diversity: float) -> np.ndarray:
        """改进的变异操作：自适应多项式变异，根据多样性调整变异强度"""
        mutated = individual.copy()
        n = len(mutated)
        eta_m = 15  # 变异分布指数
        
        # 当多样性低时增加变异率和变异强度
        adj_mutation_rate = self.mutation_rate * (1.5 if diversity < self.diversity_threshold else 1.0)
        
        for i in range(n):
            if np.random.random() < adj_mutation_rate:
                # 多项式变异
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                
                # 根据多样性调整变异步长
                step_size = self.mutation_scale * (1.5 if diversity < self.diversity_threshold else 1.0)
                mutated[i] += delta * step_size
                mutated[i] = np.clip(mutated[i], 0, 1)
        
        return mutated

    def adaptive_temperature(self, diversity: float, temperature: float) -> float:
        """根据种群多样性自适应调整温度"""
        if diversity < self.diversity_threshold * 0.7:
            # 多样性过低，提高温度促进探索
            return temperature * 1.1
        elif diversity > self.diversity_threshold * 1.3:
            # 多样性过高，降低温度促进收敛
            return temperature * 0.9
        else:
            # 正常衰减
            return temperature * self.alpha * self.beta

    def boltzmann_acceptance(self, delta_f: float, temperature: float) -> bool:
        """改进的Boltzmann接受准则，增加对劣质解的接受概率以增强探索"""
        if delta_f > 0:
            return True
        # 增加对较差解的接受概率，增强跳出局部最优的能力
        return np.random.random() < np.exp(delta_f / (temperature + 1e-10)) * 1.2

    def restart_population(self, best_individual: np.ndarray) -> np.ndarray:
        """增强的种群重启策略：保留多个优秀个体，引入多样化新个体"""
        self.restart_count += 1
        logger.info(f"种群重启，第{self.restart_count}次重启")
        
        # 保留多个优秀个体
        num_retain = min(5, self.elite_size)
        new_pop = np.zeros((self.pop_size, 8))
        
        # 保留历史上表现最好的个体
        if len(self.history_best) > 0:
            # 选择历史上最好的几个个体
            retained = self.history_best[:num_retain]
            for i, ind in enumerate(retained):
                new_pop[i] = ind.copy()
            
            # 对保留的个体进行变异，增加多样性
            for i in range(num_retain):
                new_pop[i] = self.mutation(new_pop[i], 0)  # 强制高变异率
            
            # 生成剩余个体
            remaining = self.pop_size - num_retain
            new_pop[num_retain:] = self.initialize_population()[:remaining]
        else:
            # 如果没有历史记录，生成全新种群但保留当前最优
            new_pop = self.initialize_population()
            new_pop[0] = best_individual.copy()
            
        return new_pop

    def immigration(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """移民机制：定期引入新个体，增加种群多样性"""
        new_pop = population.copy()
        num_immigrants = int(self.pop_size * self.immigration_rate)
        
        if num_immigrants > 0:
            # 替换适应度最低的个体
            worst_indices = np.argsort(fitness_values)[:num_immigrants]
            new_individuals = self.initialize_population()[:num_immigrants]
            new_pop[worst_indices] = new_individuals
            
        return new_pop

    def is_stuck_in_cycle(self) -> bool:
        """检查是否陷入循环：最近几代最优解变化很小"""
        if len(self.history_best) < self.history_window:
            return False
            
        # 检查最近history_window个最优解的差异
        recent = self.history_best[-self.history_window:]
        diffs = [np.linalg.norm(recent[i] - recent[i+1]) for i in range(len(recent)-1)]
        return np.mean(diffs) < 1e-5

    """计算个体适应度（阻断时间）- 保持不变"""
    def fitness(self, x: np.ndarray) -> float:
        """计算个体适应度（阻断时间）"""
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
        # 将比例参数转换为实际参数（8个优化变量）
        # x[0]: 速度参数
        # x[1]: 角度参数
        # x[2],x[3]: 第一个烟雾弹的投放和爆炸时间参数
        # x[4],x[5]: 第二个烟雾弹的投放和爆炸时间参数
        # x[6],x[7]: 第三个烟雾弹的投放和爆炸时间参数
        
        vi = 105 + np.array([x[0] * 70 - 35])  # 速度范围:70-140
        theta = theta_int_11 + np.array([(x[1] * 30 - 15) * np.pi / 180])  # 角度范围:-15°至15°
        
        # 调整时间参数，确保三个烟雾弹有足够的时间间隔和作用窗口
        delta_t1 = 15  # 第一个烟雾弹投放时间范围扩大
        delta_t2 = 30  # 第二个烟雾弹投放时间范围
        delta_t3 = 30  # 第三个烟雾弹投放时间范围
        
        # 第一个烟雾弹
        tdrop_11 = x[2] * delta_t1  # 投放时间:0-15
        texpl_11 = tdrop_11 + x[3] * 20  # 爆炸时间:投放后0-20秒
        
        # 第二个烟雾弹（确保在第一个之后投放）
        tdrop_12 = tdrop_11 + 1 + x[4] * delta_t2  # 投放时间:第一个+1秒后，再0-30秒
        texpl_12 = tdrop_12 + x[5] * 20  # 爆炸时间:投放后0-20秒
        
        # 第三个烟雾弹（确保在第二个之后投放）
        tdrop_13 = tdrop_12 + 1 + x[6] * delta_t3  # 投放时间:第二个+1秒后，再0-30秒
        texpl_13 = tdrop_13 + x[7] * 20  # 爆炸时间:投放后0-20秒
        
        # 整理投放和爆炸时间数组（3个烟雾弹）
        tdrop = np.array([[tdrop_11, tdrop_12, tdrop_13]])
        texpl = np.array([[texpl_11, texpl_12, texpl_13]])

        # 创建烟雾系统实例
        cover = cover_system(vi, theta, tdrop, texpl)

        # 计算阻断时间 - 提高采样精度
        pace = 300  # 提高采样精度，确保捕捉所有烟雾作用时间
        t_block = 0
        delta_t = 67 / pace  # 延长时间范围至80秒，确保覆盖所有烟雾作用时间

        for t in np.linspace(0, 67, pace):
            Mj, smokes_location = cover(t, 1)  # 只检查第一枚导弹
            if len(smokes_location) > 0 and self.checker.check(Mj, smokes_location):
                t_block += delta_t

        return t_block

    def _negative_fitness(self, x: np.ndarray) -> float:
        """用于局部优化的负适应度（因为牛顿法寻找最小值）"""
        return -self.fitness(x)

    def _gradient(self, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """计算梯度（8个变量）"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_plus = np.clip(x_plus, 0, 1)
            
            x_minus = x.copy()
            x_minus[i] -= epsilon
            x_minus = np.clip(x_minus, 0, 1)
            
            grad[i] = (self._negative_fitness(x_plus) - self._negative_fitness(x_minus)) / (2 * epsilon)
        return grad

    def _hessian(self, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """计算海森矩阵（8x8）"""
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    x_plus = x.copy()
                    x_plus[i] += epsilon
                    x_plus = np.clip(x_plus, 0, 1)
                    
                    x_minus = x.copy()
                    x_minus[i] -= epsilon
                    x_minus = np.clip(x_minus, 0, 1)
                    
                    hess[i, j] = (self._negative_fitness(x_plus) - 2 * self._negative_fitness(x) + 
                                 self._negative_fitness(x_minus)) / (epsilon **2)
                else:
                    x_ij_plus = x.copy()
                    x_ij_plus[i] += epsilon
                    x_ij_plus[j] += epsilon
                    x_ij_plus = np.clip(x_ij_plus, 0, 1)
                    
                    x_i_plus_j_minus = x.copy()
                    x_i_plus_j_minus[i] += epsilon
                    x_i_plus_j_minus[j] -= epsilon
                    x_i_plus_j_minus = np.clip(x_i_plus_j_minus, 0, 1)
                    
                    x_i_minus_j_plus = x.copy()
                    x_i_minus_j_plus[i] -= epsilon
                    x_i_minus_j_plus[j] += epsilon
                    x_i_minus_j_plus = np.clip(x_i_minus_j_plus, 0, 1)
                    
                    x_ij_minus = x.copy()
                    x_ij_minus[i] -= epsilon
                    x_ij_minus[j] -= epsilon
                    x_ij_minus = np.clip(x_ij_minus, 0, 1)
                    
                    hess[i, j] = (self._negative_fitness(x_ij_plus) - self._negative_fitness(x_i_plus_j_minus) -
                                 self._negative_fitness(x_i_minus_j_plus) + self._negative_fitness(x_ij_minus)) / (4 * epsilon**2)
                    hess[j, i] = hess[i, j]  # 对称矩阵
        return hess

    def _line_search(self, x: np.ndarray, direction: np.ndarray, grad: np.ndarray, 
                    alpha: float = 1.0, c1: float = 1e-4, c2: float = 0.9, max_iter: int = 20) -> float:
        """强 Wolfe条件线搜索"""
        f0 = self._negative_fitness(x)
        g0 = np.dot(grad, direction)
        
        for _ in range(max_iter):
            x_new = x + alpha * direction
            x_new = np.clip(x_new, 0, 1)
            f_new = self._negative_fitness(x_new)
            
            # 第一个条件：充分下降
            if f_new > f0 + c1 * alpha * g0:
                alpha *= 0.5
                continue
                
            # 计算新梯度
            grad_new = self._gradient(x_new)
            g_new = np.dot(grad_new, direction)
            
            # 第二个条件：曲率条件
            if g_new < c2 * g0:
                alpha *= 2.0
                continue
                
            return alpha
            
        return alpha  # 达到最大迭代次数

    def newton_optimize(self, x0: np.ndarray) -> np.ndarray:
        """改进的牛顿法局部优化，增加线搜索和信赖域"""
        x = x0.copy()
        for i in range(self.newton_max_iter):
            grad = self._gradient(x)
            if np.linalg.norm(grad) < self.newton_tol:
                logger.info(f"牛顿法在第{i+1}次迭代收敛")
                break
                
            hess = self._hessian(x)
            
            # 确保海森矩阵正定
            min_eig = np.min(np.real(np.linalg.eigvals(hess)))
            if min_eig <= 0:
                hess += (-min_eig + 1e-6) * np.eye(hess.shape[0])
                
            try:
                # 计算牛顿方向
                direction = -np.linalg.solve(hess, grad)
                
                # 线搜索确定步长
                alpha = self._line_search(x, direction, grad)
                x_new = x + alpha * direction
                
                # 边界约束
                x_new = np.clip(x_new, 0, 1)
                
                # 检查是否改进
                if self._negative_fitness(x_new) < self._negative_fitness(x):
                    x = x_new
                else:
                    # 尝试更小的步长
                    for _ in range(3):
                        alpha *= 0.5
                        x_new = x + alpha * direction
                        x_new = np.clip(x_new, 0, 1)
                        if self._negative_fitness(x_new) < self._negative_fitness(x):
                            x = x_new
                            break

                    else:
                        logger.info(f"牛顿法在第{i+1}次迭代未找到更优点，停止迭代")
                        break
            except np.linalg.LinAlgError:
                logger.warning("海森矩阵求解失败，使用梯度下降代替")
                x_new = x - 0.01 * grad
                x_new = np.clip(x_new, 0, 1)
                x = x_new
                
        return x

    def optimize(self, generations: int = 200) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """执行优化（增强版AE-SAGA全局搜索 + 改进牛顿法局部搜索）"""
        # 初始化种群（8维变量）
        population = self.initialize_population()
        fitness_values = np.array([self.fitness(ind) for ind in population])
        self.population = population  # 保存种群引用用于多样性计算

        # 处理可能的零适应度值，避免后续计算问题
        if np.all(fitness_values == 0):
            # 如果所有适应度都是0，添加微小差异以确保选择可以进行
            fitness_values += np.random.normal(0, 1e-8, size=fitness_values.shape)

        if self.T0 is None:
            # 更高的初始温度，促进探索
            fitness_range = np.max(fitness_values) - np.min(fitness_values)
            self.T0 = fitness_range * 10 if fitness_range > 0 else 10  # 处理所有适应度相同的情况

        T = self.T0
        best_fitness_history = []
        avg_fitness_history = []
        stagnation_count = 0
        prev_best_fitness = -np.inf
        global_best_fitness = -np.inf
        global_best_individual = None

        for generation in range(generations):
            best_idx = np.argmax(fitness_values)
            best_fitness = fitness_values[best_idx]
            best_individual = population[best_idx].copy()
            avg_fitness = np.mean(fitness_values)

            # 记录历史最优解
            self.history_best.append(best_individual)
            if len(self.history_best) > self.history_window * 2:
                self.history_best = self.history_best[-self.history_window * 2:]

            # 更新全局最优
            if best_fitness > global_best_fitness:
                global_best_fitness = best_fitness
                global_best_individual = best_individual.copy()
                logger.info(f"找到新的全局最优解: {best_fitness:.6f}")

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # 检查停滞
            if best_fitness <= prev_best_fitness + self.restart_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best_fitness = best_fitness

            # 计算多样性
            diversity = self.population_diversity()

            logger.info(f"第{generation + 1}代 - 最佳适应度: {best_fitness:.6f}, 平均适应度: {avg_fitness:.6f}, "
                        f"温度: {T:.6f}, 多样性: {diversity:.6f}, 停滞计数: {stagnation_count}")

            # 检查终止条件
            if T < self.T_final:
                logger.info(f"温度低于阈值，AE-SAGA终止于第{generation + 1}代")
                break

            # 种群重启机制：更激进的重启策略
            if (stagnation_count >= self.max_stagnation or 
                self.is_stuck_in_cycle() or 
                (diversity < self.diversity_threshold * 0.5 and generation > 10)) and self.restart_count < self.max_restarts:
                population = self.restart_population(best_individual)
                fitness_values = np.array([self.fitness(ind) for ind in population])
                
                # 处理可能的零适应度值
                if np.all(fitness_values == 0):
                    fitness_values += np.random.normal(0, 1e-8, size=fitness_values.shape)
                    
                stagnation_count = 0  # 重置停滞计数
                # 重启后温度调整，比之前略低但仍保持探索能力
                T = self.T0 * (0.6 ** self.restart_count)
                self.population = population
                continue

            # 构建新种群
            # 保留精英个体
            elite_indices = np.argsort(fitness_values)[-self.elite_size:]
            new_population = population[elite_indices].tolist()

            # 选择操作
            selected_indices = self.selection(population, fitness_values)

            # 交叉和变异
            for i in range(0, len(selected_indices), 2):
                if i + 1 >= len(selected_indices):
                    # 处理奇数情况
                    idx = selected_indices[i]
                    new_population.append(population[idx].copy())
                    break

                idx1, idx2 = selected_indices[i], selected_indices[i + 1]
                parent1, parent2 = population[idx1], population[idx2]
                
                # 交叉操作
                child1, child2 = self.crossover(parent1, parent2)

                # 基于当前多样性的变异
                child1 = self.mutation(child1, diversity)
                child2 = self.mutation(child2, diversity)

                # 评估子代
                f_child1 = self.fitness(child1)
                f_child2 = self.fitness(child2)
                f_parent1 = fitness_values[idx1]
                f_parent2 = fitness_values[idx2]

                # Boltzmann接受
                if self.boltzmann_acceptance(f_child1 - f_parent1, T):
                    new_population.append(child1)
                else:
                    new_population.append(parent1.copy())

                if self.boltzmann_acceptance(f_child2 - f_parent2, T):
                    new_population.append(child2)
                else:
                    new_population.append(parent2.copy())

            # 保持种群大小
            if len(new_population) > self.pop_size:
                new_population = new_population[:self.pop_size]
            elif len(new_population) < self.pop_size:
                while len(new_population) < self.pop_size:
                    new_population.append(np.random.random(8))  # 8维变量

            # 定期引入移民增加多样性
            if generation % 10 == 0:  # 每10代引入一次移民
                new_population = self.immigration(np.array(new_population), 
                                                  np.array([self.fitness(ind) for ind in new_population]))

            population = np.array(new_population)
            self.population = population
            fitness_values = np.array([self.fitness(ind) for ind in population])
            
            # 处理可能的零适应度值
            if np.all(fitness_values == 0):
                fitness_values += np.random.normal(0, 1e-8, size=fitness_values.shape)
            
            # 自适应温度调整
            T = self.adaptive_temperature(diversity, T)

        # 多起点牛顿法局部搜索，增加起点数量以找到更好的局部最优
        logger.info("开始多起点牛顿法局部优化...")
        
        # 选择更多候选个体进行局部优化
        sorted_indices = np.argsort(fitness_values)[::-1]
        # 选择更多候选解，包括一些中等表现的解以增加多样性
        num_candidates = min(8, self.pop_size//2)
        candidates = population[sorted_indices[:num_candidates]]
        candidates = np.append(candidates, [global_best_individual], axis=0)
        
        # 加入一些经过轻微变异的候选解，扩大搜索范围
        mutated_candidates = [self.mutation(global_best_individual, 0.1) for _ in range(2)]
        candidates = np.append(candidates, mutated_candidates, axis=0)
        
        best_refined_fitness = -np.inf
        best_refined_individual = None
        
        for i, candidate in enumerate(candidates):
            logger.info(f"对第{i+1}个候选解进行牛顿法优化")
            refined = self.newton_optimize(candidate)
            refined_fitness = self.fitness(refined)
            
            if refined_fitness > best_refined_fitness:
                best_refined_fitness = refined_fitness
                best_refined_individual = refined

        # 比较局部优化前后结果
        if best_refined_fitness > global_best_fitness:
            best_fitness_history.append(best_refined_fitness)
            avg_fitness_history.append(avg_fitness_history[-1])  # 保持长度一致
            logger.info(f"牛顿法优化成功，适应度从{global_best_fitness:.6f}提升至{best_refined_fitness:.6f}")
            best_individual = best_refined_individual
            best_fitness = best_refined_fitness
        else:
            logger.info("牛顿法未找到更优解，保留全局搜索结果")
            best_individual = global_best_individual
            best_fitness = global_best_fitness

        return best_individual, best_fitness, best_fitness_history, avg_fitness_history


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 使用增强版优化器
    optimizer = EnhancedSmokeOptimizer()
    best_ind, best_fit, best_history, avg_history = optimizer.optimize(generations=200)
    
    # 1. 结果可视化
    plt.figure(figsize=(12, 6))
    plt.plot(best_history, label='最佳适应度（阻断时间）', color='red', linewidth=2)
    plt.plot(avg_history, label='平均适应度', color='blue', linestyle='--', linewidth=1.5)
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值（秒）')
    plt.title('优化过程中适应度变化曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('enhanced_optimization_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 参数反标准化（还原为实际物理参数）
    def denormalize_params(x):
        """将标准化参数转换为实际物理参数"""
        # 计算初始角度（与原代码中fitness函数保持一致）
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
        
        # 反标准化计算
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
    
    # 输出反标准化后的参数
    actual_params = denormalize_params(best_ind)
    print("\n===== 优化结果 =====")
    print(f"最佳阻断时间: {best_fit:.6f} 秒")
    print("\n===== 实际物理参数 =====")
    for name, value in actual_params.items():
        print(f"{name}: {value:.6f}")
    print("\n===== 标准化参数 (0-1) =====")
    print(f"最佳参数 (8个变量): {best_ind}")
    print("\n参数解释:")
    print("x0: 速度参数, x1: 角度参数")
    print("x2,x3: 第一个烟雾弹投放和爆炸时间参数")
    print("x4,x5: 第二个烟雾弹投放和爆炸时间参数")
    print("x6,x7: 第三个烟雾弹投放和爆炸时间参数")
    