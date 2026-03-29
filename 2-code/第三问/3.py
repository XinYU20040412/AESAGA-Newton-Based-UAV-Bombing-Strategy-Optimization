import numpy as np
import logging
from typing import Tuple, List
from system_at_t import cover_system
from cover_checker import AdvancedMissileSmokeChecker

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmokeOptimizer:
    def __init__(self, pop_size: int = 100, elite_size: int = 10, 
                 alpha: float = 0.95, beta: float = 1.0, T_final: float = 1e-8,
                 restart_threshold: float = 1e-6, max_stagnation: int = 20,
                 max_restarts: int = 3, diversity_threshold: float = 0.5,
                 newton_max_iter: int = 50, newton_tol: float = 1e-6):
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.alpha = alpha  # 温度衰减系数
        self.beta = beta    # 自适应调整因子
        self.T_final = T_final  # 终止温度
        self.T0 = None      # 初始温度，将在优化开始时计算
        self.restart_threshold = restart_threshold  # 停滞判断阈值
        self.max_stagnation = max_stagnation        # 最大停滞代数
        self.restart_count = 0                      # 重启计数
        self.max_restarts = max_restarts            # 最大重启次数
        self.diversity_threshold = diversity_threshold  # 多样性阈值
        self.population = None  # 当前种群
        self.checker = AdvancedMissileSmokeChecker()  # 烟雾遮挡检查器
        
        # 牛顿法参数
        self.newton_max_iter = newton_max_iter
        self.newton_tol = newton_tol

    def initialize_population(self) -> np.ndarray:
        """初始化种群，8个优化变量"""
        return np.random.random((self.pop_size, 8))  # 确保是8维变量

    def population_diversity(self) -> float:
        """计算种群多样性"""
        if self.population is None or len(self.population) == 0:
            return 0.0
        return np.mean(np.std(self.population, axis=0))

    def selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """选择操作：基于适应度的轮盘赌选择"""
        # 确保适应度为正
        min_fitness = np.min(fitness_values)
        if min_fitness <= 0:
            adjusted_fitness = fitness_values - min_fitness + 1e-6
        else:
            adjusted_fitness = fitness_values
        
        # 计算选择概率
        probs = adjusted_fitness / np.sum(adjusted_fitness)
        # 选择与种群大小匹配的个体（减去精英数量）
        selected_size = self.pop_size - self.elite_size
        selected_indices = np.random.choice(len(population), size=selected_size, p=probs)
        return selected_indices

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, pc: float) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作：模拟二进制交叉"""
        if np.random.random() < pc:
            child1, child2 = parent1.copy(), parent2.copy()
            # 随机选择交叉点
            for i in range(len(parent1)):
                if np.random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutation(self, individual: np.ndarray, pm: float) -> np.ndarray:
        """变异操作：多项式变异"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < pm:
                # 多项式变异
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (20 + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (20 + 1))
                mutated[i] = np.clip(mutated[i] + delta, 0, 1)
        return mutated

    def adaptive_Pc(self, f: float, avg_f: float, best_f: float) -> float:
        """自适应交叉概率"""
        if f > avg_f:
            return 0.9 - 0.4 * (best_f - f) / (best_f - avg_f + 1e-10)
        else:
            return 0.5

    def adaptive_Pm(self, f: float, avg_f: float, best_f: float) -> float:
        """自适应变异概率"""
        if f > avg_f:
            return 0.1 - 0.05 * (best_f - f) / (best_f - avg_f + 1e-10)
        else:
            return 0.1

    def boltzmann_acceptance(self, delta_f: float, temperature: float) -> bool:
        """Boltzmann接受准则"""
        if delta_f > 0:
            return True
        return np.random.random() < np.exp(delta_f / temperature)

    def restart_population(self, best_individual: np.ndarray) -> np.ndarray:
        """种群重启：保留最优个体，其余随机生成"""
        self.restart_count += 1
        logger.info(f"种群重启，第{self.restart_count}次重启")
        new_pop = np.random.random((self.pop_size, 8))  # 8维变量
        new_pop[0] = best_individual  # 保留最优个体
        return new_pop

    """计算个体适应度（阻断时间）"""
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

    """改进的牛顿法局部优化，增加线搜索和信赖域"""
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

    """执行优化（增强版AE-SAGA全局搜索 + 改进牛顿法局部搜索）"""
    def optimize(self, generations: int = 150) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """执行优化（增强版AE-SAGA全局搜索 + 改进牛顿法局部搜索）"""
        # 初始化种群（8维变量）
        population = self.initialize_population()
        fitness_values = np.array([self.fitness(ind) for ind in population])
        self.population = population  # 保存种群引用用于多样性计算

        if self.T0 is None:
            self.T0 = (np.max(fitness_values) - np.min(fitness_values)) * 5  # 更高的初始温度

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

            # 更新全局最优
            if best_fitness > global_best_fitness:
                global_best_fitness = best_fitness
                global_best_individual = best_individual.copy()

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # 检查停滞
            if best_fitness <= prev_best_fitness + self.restart_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best_fitness = best_fitness

            # 自适应调整beta和温度
            diversity = self.population_diversity()
            if generation > 5:
                # 根据多样性调整beta
                if diversity < self.diversity_threshold * 0.5:
                    self.beta = min(1.2, self.beta * 1.05)  # 多样性低时减缓温度下降
                elif diversity > self.diversity_threshold * 1.5:
                    self.beta = max(0.8, self.beta * 0.95)  # 多样性高时加快温度下降

            logger.info(f"第{generation + 1}代 - 最佳适应度: {best_fitness:.6f}, 平均适应度: {avg_fitness:.6f}, "
                        f"温度: {T:.6f}, 多样性: {diversity:.6f}, 停滞计数: {stagnation_count}")

            # 检查终止条件
            if T < self.T_final:
                logger.info(f"温度低于阈值，AE-SAGA终止于第{generation + 1}代")
                break

            # 种群重启机制
            if stagnation_count >= self.max_stagnation and self.restart_count < self.max_restarts:
                population = self.restart_population(best_individual)
                fitness_values = np.array([self.fitness(ind) for ind in population])
                stagnation_count = 0  # 重置停滞计数
                T = self.T0 * (0.5 ** self.restart_count)  # 重启后温度降低
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
                f1, f2 = fitness_values[idx1], fitness_values[idx2]

                # 自适应交叉
                Pc = self.adaptive_Pc(max(f1, f2), avg_fitness, best_fitness)
                child1, child2 = self.crossover(parent1, parent2, Pc)

                # 自适应变异
                Pm1 = self.adaptive_Pm(f1, avg_fitness, best_fitness)
                Pm2 = self.adaptive_Pm(f2, avg_fitness, best_fitness)
                child1 = self.mutation(child1, Pm1)
                child2 = self.mutation(child2, Pm2)

                # 评估子代
                f_child1 = self.fitness(child1)
                f_child2 = self.fitness(child2)

                # Boltzmann接受
                if self.boltzmann_acceptance(f_child1 - f1, T):
                    new_population.append(child1)
                else:
                    new_population.append(parent1)

                if self.boltzmann_acceptance(f_child2 - f2, T):
                    new_population.append(child2)
                else:
                    new_population.append(parent2)

            # 保持种群大小
            if len(new_population) > self.pop_size:
                new_population = new_population[:self.pop_size]
            elif len(new_population) < self.pop_size:
                while len(new_population) < self.pop_size:
                    new_population.append(np.random.random(8))  # 确保是8维变量

            # 偶尔引入随机个体增加多样性
            if np.random.random() < 0.05:  # 5%概率
                random_idx = np.random.randint(0, self.pop_size)
                new_population[random_idx] = np.random.random(8)  # 确保是8维变量

            population = np.array(new_population)
            self.population = population
            fitness_values = np.array([self.fitness(ind) for ind in population])
            
            # 温度衰减
            T *= self.alpha * self.beta

        # 多起点牛顿法局部搜索
        logger.info("开始多起点牛顿法局部优化...")
        
        # 选择多个优秀个体进行局部优化
        sorted_indices = np.argsort(fitness_values)[::-1]
        candidates = population[sorted_indices[:min(5, self.pop_size//2)]]  # 选择前5个优秀个体
        candidates = np.append(candidates, [global_best_individual], axis=0)  # 加入全局最优
        
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


import matplotlib.pyplot as plt  # 新增：导入可视化库
from typing import Tuple, List


# （中间代码保持不变，省略重复部分）

if __name__ == "__main__":
    optimizer = SmokeOptimizer()
    best_ind, best_fit, best_history, avg_history = optimizer.optimize(generations=150)
    
    # 1. 结果可视化
    plt.figure(figsize=(12, 6))
    plt.plot(best_history, label='最佳适应度（阻断时间）', color='red', linewidth=2)
    plt.plot(avg_history, label='平均适应度', color='blue', linestyle='--', linewidth=1.5)
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值（秒）')
    plt.title('优化过程中适应度变化曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('optimization_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 参数反标准化（还原为实际物理意义的参数）
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
    []