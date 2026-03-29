import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import logging
from cover_checker import AdvancedMissileSmokeChecker
from system_at_t import cover_system

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
min_theta = 25
delta_trop = 60
delta_t_expl_diff = 20
theta_int = initjude(2, 2)
class AESAGAWithNewton:
    """
    结合AE-SAGA全局搜索与牛顿法局部搜索的混合优化算法
    增强了跳出局部最优的能力
    """

    def __init__(self,
                 pop_size: int = 50,
                 elite_size: float = 0.1,
                 Pc1: float = 0.95,
                 Pc2: float = 0.4,
                 Pm1: float = 0.3,
                 Pm2: float = 0.001,
                 T0: float = 200,
                 T_final: float = 0.001,
                 alpha: float = 0.97,  # 减缓温度衰减
                 beta: float = 1.0,
                 max_stagnation: int = 8,  # 增加停滞容忍度
                 newton_max_iter: int = 40,  # 增加牛顿法迭代次数
                 newton_tol: float = 1e-5,  # 提高牛顿法精度
                 restart_threshold: float = 0.001,  # 重启阈值
                 diversity_threshold: float = 0.1):  # 多样性阈值
        """初始化算法参数"""
        self.pop_size = pop_size
        self.elite_size = int(pop_size * elite_size)
        self.Pc1 = Pc1
        self.Pc2 = Pc2
        self.Pm1 = Pm1
        self.Pm2 = Pm2
        self.T0 = T0
        self.T_final = T_final
        self.alpha = alpha
        self.beta = beta
        self.max_stagnation = max_stagnation
        self.restart_threshold = restart_threshold
        self.diversity_threshold = diversity_threshold
        
        # 牛顿法参数
        self.newton_max_iter = newton_max_iter
        self.newton_tol = newton_tol
        self.epsilon = 1e-6  # 更小的数值微分步长，提高精度
        
        # 新增参数
        self.restart_count = 0
        self.max_restarts = 3  # 最大重启次数
        self.line_search_alpha = 0.5  # 线搜索步长因子
        self.line_search_beta = 0.8   # 线搜索收缩因子

        # 创建烟雾检查器实例
        self.checker = AdvancedMissileSmokeChecker()

    def fitness(self, x: np.ndarray) -> float:
        """计算个体适应度（阻断时间）"""
        
        # 将比例参数转换为实际参数
         # 转换为实际参数
        
        vi =  np.array([0,0,x[0] * (70) +70])
        theta =  np.array([0,0,theta_int +((x[1] * 2*min_theta-min_theta) * np.pi / 180) ])
        tdrop = np.array([[100000,100000,100000],[100000,100000,100000],[x[2] *delta_trop,100000,100000]])
        texpl = np.array([[100000,100000,100000],[100000,100000,100000],[x[3] * delta_t_expl_diff + x[2] * delta_trop,100000,100000]])

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

    def _negative_fitness(self, x: np.ndarray) -> float:
        """用于牛顿法的负适应度函数（将最大化问题转为最小化）"""
        return -self.fitness(x)

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        """数值计算梯度，使用中心差分提高精度"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += self.epsilon
            x_minus = x.copy()
            x_minus[i] -= self.epsilon
            grad[i] = (self._negative_fitness(x_plus) - self._negative_fitness(x_minus)) / (2 * self.epsilon)
        return grad

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        """数值计算海森矩阵，增加正则化处理"""
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += self.epsilon
            x_minus = x.copy()
            x_minus[i] -= self.epsilon
            grad_plus = self._gradient(x_plus)
            grad_minus = self._gradient(x_minus)
            hess[i] = (grad_plus - grad_minus) / (2 * self.epsilon)
        
        # 增加正则化，提高数值稳定性
        min_eig = np.min(np.real(np.linalg.eigvals(hess)))
        if min_eig < 1e-6:
            hess += (1e-6 - min_eig) * np.eye(n)
        return hess

    def _line_search(self, x: np.ndarray, direction: np.ndarray, grad: np.ndarray) -> float:
        """线搜索寻找最佳步长"""
        alpha = 1.0  # 初始步长
        f_x = self._negative_fitness(x)
        x_new = x + alpha * direction
        
        # 确保在搜索空间内
        if np.any(x_new < 0) or np.any(x_new > 1):
            alpha = 0.1
            
        # Armijo条件
        while self._negative_fitness(x_new) > f_x + self.line_search_alpha * alpha * np.dot(grad, direction):
            alpha *= self.line_search_beta
            x_new = x + alpha * direction
            if alpha < 1e-8:  # 步长过小
                return 1e-8
        return alpha

    def newton_optimize(self, x0: np.ndarray) -> np.ndarray:
        """改进的牛顿法局部优化，增加线搜索和信赖域"""
        x = x0.copy()
        for i in range(self.newton_max_iter):
            grad = self._gradient(x)
            if np.linalg.norm(grad) < self.newton_tol:
                logger.info(f"牛顿法在第{i+1}次迭代收敛")
                break
                
            hess = self._hessian(x)
            
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

    def adaptive_Pc(self, f_prime: float, f_avg: float, f_max: float) -> float:
        """改进的自适应交叉概率计算"""
        if f_max == f_avg:
            return self.Pc1
        # 当种群多样性低时增加交叉概率
        diversity = self.population_diversity()
        diversity_factor = 1.0 + (self.diversity_threshold - min(diversity, self.diversity_threshold)) / self.diversity_threshold
        
        if f_prime >= f_avg:
            return diversity_factor * (self.Pc1 - (self.Pc1 - self.Pc2) * (f_prime - f_avg) / (f_max - f_avg))
        else:
            return diversity_factor * self.Pc1

    def adaptive_Pm(self, f: float, f_avg: float, f_max: float) -> float:
        """改进的自适应变异概率计算"""
        if f_max == f_avg:
            return self.Pm1
            
        # 当种群多样性低时增加变异概率
        diversity = self.population_diversity()
        diversity_factor = 1.0 + (self.diversity_threshold - min(diversity, self.diversity_threshold)) / self.diversity_threshold
        
        if f >= f_avg:
            return diversity_factor * (self.Pm1 - (self.Pm1 - self.Pm2) * (f_max - f) / (f_max - f_avg))
        else:
            return diversity_factor * self.Pm1

    def population_diversity(self) -> float:
        """计算种群多样性"""
        if not hasattr(self, 'population'):
            return 1.0
        # 计算个体间平均欧氏距离
        distances = []
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                distances.append(np.linalg.norm(self.population[i] - self.population[j]))
        return np.mean(distances) if distances else 1.0

    def initialize_population(self) -> np.ndarray:
        """初始化种群，使用拉丁超立方抽样提高初始多样性"""
        n_dim = 4
        pop = np.zeros((self.pop_size, n_dim))
        for i in range(n_dim):
            # 拉丁超立方抽样
            pop[:, i] = np.random.permutation(np.linspace(0, 1, self.pop_size))
            # 添加随机扰动
            pop[:, i] += np.random.uniform(-0.05, 0.05, self.pop_size)
        return np.clip(pop, 0, 1)

    def selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """改进的选择算子，结合轮盘赌和锦标赛选择"""
        # 前30%使用锦标赛选择，保持优秀个体
        sorted_indices = np.argsort(fitness_values)[::-1]
        top_indices = sorted_indices[:int(0.3 * len(population))]
        
        # 剩余使用轮盘赌选择
        remaining_indices = sorted_indices[int(0.3 * len(population)):]
        min_fitness = np.min(fitness_values[remaining_indices])
        adjusted_fitness = fitness_values[remaining_indices] - min_fitness + 1e-6 if min_fitness < 0 else fitness_values[remaining_indices]
        fitness_sum = np.sum(adjusted_fitness)
        selection_probs = adjusted_fitness / fitness_sum if fitness_sum != 0 else np.ones(len(adjusted_fitness)) / len(adjusted_fitness)
        
        # 组合选择结果
        tournament_size = 3
        tournament_selected = []
        for _ in range(self.elite_size):
            candidates = np.random.choice(top_indices, tournament_size)
            tournament_selected.append(candidates[np.argmax(fitness_values[candidates])])
            
        roulette_selected = np.random.choice(remaining_indices, 
                                           size=len(population) - self.elite_size - len(tournament_selected),
                                           p=selection_probs, 
                                           replace=True)
                                           
        return np.concatenate([tournament_selected, roulette_selected])

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, Pc: float) -> Tuple[np.ndarray, np.ndarray]:
        """自适应交叉算子，根据父母相似度选择交叉方式"""
        if np.random.random() > Pc:
            return parent1.copy(), parent2.copy()
            
        # 计算父母相似度
        similarity = 1 - np.linalg.norm(parent1 - parent2) / np.sqrt(len(parent1))
        
        if similarity > 0.7:  # 父母相似，使用均匀交叉增加多样性
            mask = np.random.random(len(parent1)) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        else:  # 父母差异大，使用单点交叉保留优良特性
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            
        return child1, child2

    def mutation(self, individual: np.ndarray, Pm: float) -> np.ndarray:
        """改进的变异算子，结合高斯变异和均匀变异"""
        mutated = individual.copy()
        n = len(mutated)
        
        for i in range(n):
            if np.random.random() < Pm:
                # 有50%概率使用高斯变异，50%使用均匀变异
                if np.random.random() < 0.5:
                    # 高斯变异，均值为当前值，标准差自适应
                    sigma = 0.1 * (1 + (self.diversity_threshold - min(self.population_diversity(), self.diversity_threshold)) / self.diversity_threshold)
                    mutated[i] += np.random.normal(0, sigma)
                else:
                    # 均匀变异
                    mutated[i] = np.random.random()
                    
        return np.clip(mutated, 0, 1)

    def boltzmann_acceptance(self, delta_f: float, T: float) -> bool:
        """改进的Boltzmann接受准则，随温度动态调整"""
        if delta_f > 0:
            return True
        # 当温度高时接受更多差解，温度低时更严格
        acceptance_prob = np.exp(delta_f / (T + 1e-8))
        # 随着迭代增加，降低对差解的接受概率
        return np.random.random() < acceptance_prob

    def restart_population(self, best_individual: np.ndarray) -> np.ndarray:
        """种群重启策略，保留最优个体并重新生成其他个体"""
        self.restart_count += 1
        logger.info(f"种群重启，第{self.restart_count}次重启")
        new_pop = np.zeros((self.pop_size, len(best_individual)))
        # 保留最优个体
        new_pop[0] = best_individual.copy()
        # 围绕最优个体生成一些相似个体
        for i in range(1, int(self.pop_size * 0.3)):
            new_pop[i] = best_individual + np.random.normal(0, 0.05, len(best_individual))
        # 生成全新个体
        for i in range(int(self.pop_size * 0.3), self.pop_size):
            new_pop[i] = np.random.random(len(best_individual))
        return np.clip(new_pop, 0, 1)

    def optimize(self, generations: int = 150) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """执行优化（增强版AE-SAGA全局搜索 + 改进牛顿法局部搜索）"""
        # 初始化种群
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
                    new_population.append(np.random.random(4))

            # 偶尔引入随机个体增加多样性
            if np.random.random() < 0.05:  # 5%概率
                random_idx = np.random.randint(0, self.pop_size)
                new_population[random_idx] = np.random.random(4)

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

    def plot_results(self, best_fitness_history: List[float], avg_fitness_history: List[float]):
        """绘制优化结果"""
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(best_fitness_history) + 1), best_fitness_history, 'r-', label='最佳适应度')
        plt.plot(range(1, len(avg_fitness_history) + 1), avg_fitness_history, 'b--', label='平均适应度')
        plt.xlabel('代数')
        plt.ylabel('适应度值 (阻断时间/秒)')
        plt.title('AE-SAGA+牛顿法优化过程 - 适应度变化')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        diff = np.array(best_fitness_history) - np.array(avg_fitness_history)
        plt.plot(range(1, len(diff) + 1), diff, 'g-', label='最佳与平均适应度差值')
        plt.xlabel('代数')
        plt.ylabel('适应度差值')
        plt.title('最佳与平均适应度差值变化')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        if hasattr(self, 'population'):
            diversity_history = [self.population_diversity()]  # 这只是示例，实际应在迭代中记录
            # 注意：实际使用时需要在optimize方法中记录多样性历史
            plt.plot(range(1, len(diversity_history) + 1), diversity_history, 'm-', label='种群多样性')
            plt.xlabel('代数')
            plt.ylabel('多样性')
            plt.title('种群多样性变化')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('aesaga_newton_optimization.png', dpi=300)
        plt.close()


# 运行优化算法
if __name__ == "__main__":
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

    optimizer = AESAGAWithNewton(
        pop_size=60,  # 增加种群规模
        elite_size=0.15,
        Pc1=0.95,
        Pc2=0.4,
        Pm1=0.3,
        Pm2=0.001,
        T0=200.0,  # 更高的初始温度
        T_final=0.01,
        alpha=0.97,
        beta=1.0,
        max_stagnation=8,
        newton_max_iter=40,
        newton_tol=1e-5,
        restart_threshold=0.001,
        diversity_threshold=0.1
    )

    best_individual, best_fitness, best_history, avg_history = optimizer.optimize(generations=150)  # 增加迭代次数

    optimizer.plot_results(best_history, avg_history)
    
        
     

    # 转换为实际参数
    best_vi = best_individual[0] * (70) + 70
    best_theta =theta_int+ (best_individual[1] * 2*min_theta-min_theta)* np.pi / 180###########################################################################################################################
    best_tdrop = best_individual[2] * delta_trop
    best_texpl_diff = best_individual[3] * delta_t_expl_diff
    best_texpl = best_tdrop + best_texpl_diff

    print("\n最佳参数:")
    print(f"速度: {best_vi:.2f} m/s")
    print(f"角度: {best_theta:.2f}°")
    print(f"释放时间: {best_tdrop:.2f} s")
    print(f"起爆时间差: {best_texpl_diff:.2f} s")
    print(f"起爆时间: {best_texpl:.2f} s")
    print(f"阻断时间: {best_fitness:.6f} s")

    with open('aesaga_newton_best_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"最佳适应度(阻断时间): {best_fitness:.6f} s\n")
        f.write(f"速度: {best_vi:.2f} m/s\n")
        f.write(f"角度: {best_theta:.2f}°\n")
        f.write(f"释放时间: {best_tdrop:.2f} s\n")
        f.write(f"起爆时间差: {best_texpl_diff:.2f} s\n")
        f.write(f"起爆时间: {best_texpl:.2f} s\n")

    print("\n优化完成！结果已保存到 'aesaga_newton_best_result.txt'")
    print("优化过程图表已保存到 'aesaga_newton_optimization.png'")