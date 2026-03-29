import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import logging
from cover_checker import AdvancedMissileSmokeChecker
from system_at_t import cover_system

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AESAGA:
    """
    自适应精英模拟退火遗传混合算法（AE-SAGA）
    结合了遗传算法的全局搜索能力和模拟退火算法的局部搜索能力
    """

    def __init__(self,
                 pop_size: int = 100,
                 elite_size: float = 0.1,
                 Pc1: float = 0.9,
                 Pc2: float = 0.6,
                 Pm1: float = 0.2,
                 Pm2: float = 0.001,
                 T0: float = None,
                 T_final: float = 0.001,
                 alpha: float = 0.95,
                 beta: float = 1.0,
                 max_stagnation: int = 10):
        """
        初始化AE-SAGA算法参数

        参数:
        pop_size: 种群大小
        elite_size: 精英比例
        Pc1, Pc2: 交叉概率参数
        Pm1, Pm2: 变异概率参数
        T0: 初始温度
        T_final: 终止温度
        alpha: 基础温度衰减系数
        beta: 解质量调节因子
        max_stagnation: 最大停滞代数
        """
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

        # 创建烟雾检查器实例
        self.checker = AdvancedMissileSmokeChecker()

    def fitness(self, x: np.ndarray) -> float:
        """
        计算个体适应度（阻断时间）

        参数:
        x: 个体基因 [速度比例, 角度比例, 释放时间比例, 起爆时间差比例]

        返回:
        适应度值（阻断时间）

        """
        def initjude(idex_FY,index_M_bomb):
            FY=[[17800,0,1800],[12000,1400,1400],[6000,-3000,700],[11000,2000,1800],[13000,-2000,1300]]
            M_int=[[20000,0,2000],[19000,600,2100],[18000,-600,1900]]
            i=idex_FY
            j=index_M_bomb
            Xfy=FY[i][0]
            Yfy=FY[i][1]
            XG=(Xfy+M_int[j][0]+0)/3
            YG=(Yfy+M_int[j][1]+200)/3
            xx=XG-Xfy
            yy=YG-Yfy
            l=(xx**2+yy**2)**0.5
            cos=xx/((xx**2+yy**2)**0.5)
            theta=np.arccos(cos)
            if yy<0:
                theta=2*np.pi-theta


            return theta
        theta_int=initjude(0,0)
        # 将比例参数转换为实际参数
        vi = 105+np.array([x[0] * (70)-35 ])
        theta = theta_int+np.array([(x[1] * (30)-15)* np.pi / 180])  # 转换为弧度
        tdrop = np.array([[x[2] * 5]])
        texpl = np.array([[x[3] * 5 +x[2]*5]])

        # 创建烟雾系统实例
        cover = cover_system(vi, theta, tdrop, texpl)

        # 减少采样点数量以提高性能
        pace = 1000
        t_block = 0
        delta_t = 67 / pace

        # 计算阻断时间
        for t in np.linspace(0, 67, pace):
            Mj, smokes_location = cover(t, 1)
            if len(smokes_location) > 0:
                if self.checker.check(Mj, smokes_location):
                    t_block += delta_t

        return t_block

    def adaptive_Pc(self, f_prime: float, f_avg: float, f_max: float) -> float:
        """
        自适应交叉概率计算

        参数:
        f_prime: 交叉个体中较大的适应度
        f_avg: 种群平均适应度
        f_max: 种群最大适应度

        返回:
        自适应交叉概率
        """
        if f_max == f_avg:  # 避免除以零
            return self.Pc1

        if f_prime >= f_avg:
            return self.Pc1 - (self.Pc1 - self.Pc2) * (f_prime - f_avg) / (f_max - f_avg)
        else:
            return self.Pc1

    def adaptive_Pm(self, f: float, f_avg: float, f_max: float) -> float:
        """
        自适应变异概率计算

        参数:
        f: 个体适应度
        f_avg: 种群平均适应度
        f_max: 种群最大适应度

        返回:
        自适应变异概率
        """
        if f_max == f_avg:  # 避免除以零
            return self.Pm1

        if f >= f_avg:
            return self.Pm1 - (self.Pm1 - self.Pm2) * (f_max - f) / (f_max - f_avg)
        else:
            return self.Pm1

    def initialize_population(self) -> np.ndarray:
        """
        初始化种群

        返回:
        初始种群
        """
        return np.random.random((self.pop_size, 4))

    def selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """
        轮盘赌选择

        参数:
        population: 种群
        fitness_values: 适应度值

        返回:
        被选中的个体索引
        """
        # 确保适应度值为正
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 1e-6

        # 计算选择概率
        fitness_sum = np.sum(fitness_values)
        if fitness_sum == 0:
            selection_probs = np.ones(len(fitness_values)) / len(fitness_values)
        else:
            selection_probs = fitness_values / fitness_sum

        # 选择个体
        selected_indices = np.random.choice(
            len(population),
            size=len(population) - self.elite_size,
            p=selection_probs,
            replace=True
        )

        return selected_indices

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, Pc: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        单点交叉

        参数:
        parent1: 父代1
        parent2: 父代2
        Pc: 交叉概率

        返回:
        两个子代
        """
        if np.random.random() > Pc:
            return parent1.copy(), parent2.copy()

        # 随机选择交叉点
        crossover_point = np.random.randint(1, 4)

        # 执行交叉
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return child1, child2

    def mutation(self, individual: np.ndarray, Pm: float) -> np.ndarray:
        """
        随机变异

        参数:
        individual: 个体
        Pm: 变异概率

        返回:
        变异后的个体
        """
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < Pm:
                mutated[i] = np.random.random()

        return np.clip(mutated, 0, 1)

    def boltzmann_acceptance(self, delta_f: float, T: float) -> bool:
        """
        Boltzmann接受准则

        参数:
        delta_f: 适应度差异 (f_new - f_old)
        T: 当前温度

        返回:
        是否接受新解
        """
        if delta_f > 0:
            return True
        else:
            return np.random.random() < np.exp(delta_f / T)

    def optimize(self, generations: int = 100) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """
        执行优化

        参数:
        generations: 迭代代数

        返回:
        最佳个体, 最佳适应度, 最佳适应度历史, 平均适应度历史
        """
        # 初始化种群
        population = self.initialize_population()

        # 计算初始适应度
        fitness_values = np.array([self.fitness(ind) for ind in population])

        # 初始化温度
        if self.T0 is None:
            self.T0 = (np.max(fitness_values) - np.min(fitness_values)) * 4  # 根据问题规模自适应

        T = self.T0

        # 初始化历史记录
        best_fitness_history = []
        avg_fitness_history = []
        best_individual_history = []

        # 初始化精英池
        elite_pool = []

        # 初始化停滞计数器
        stagnation_count = 0
        prev_best_fitness = -np.inf

        # 开始迭代
        for generation in range(generations):
            # 计算统计量
            best_idx = np.argmax(fitness_values)
            best_fitness = fitness_values[best_idx]
            best_individual = population[best_idx].copy()
            avg_fitness = np.mean(fitness_values)

            # 更新历史记录
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            best_individual_history.append(best_individual)

            # 更新精英池
            elite_indices = np.argsort(fitness_values)[-self.elite_size:]
            elite_pool = population[elite_indices].tolist()

            # 检查停滞情况
            if best_fitness <= prev_best_fitness:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best_fitness = best_fitness

            # 自适应调整beta
            if generation > 3 and stagnation_count >= 3:
                # 计算最近几代的改进率
                improvement_rate = (best_fitness_history[-1] - best_fitness_history[-4]) / 3
                if improvement_rate < 0.01 * best_fitness_history[-4]:  # 提升率<1%
                    self.beta = 0.9  # 加速降温
                elif improvement_rate > 0.05 * best_fitness_history[-4]:  # 提升率>5%
                    self.beta = 1.0  # 保持高温探索
                else:
                    self.beta = 0.95  # 正常降温

            # 输出当前代信息
            logger.info(
                f"第{generation + 1}代 - 最佳适应度: {best_fitness:.6f}, 平均适应度: {avg_fitness:.6f}, 温度: {T:.6f}")

            # 检查终止条件
            if T < self.T_final or stagnation_count >= self.max_stagnation:
                logger.info(f"优化终止于第{generation + 1}代")
                break

            # 创建新种群
            new_population = []

            # 精英保留
            new_population.extend(elite_pool)

            # 选择操作
            selected_indices = self.selection(population, fitness_values)

            # 交叉和变异
            for i in range(0, len(selected_indices), 2):
                if i + 1 >= len(selected_indices):
                    break

                # 选择父代
                idx1, idx2 = selected_indices[i], selected_indices[i + 1]
                parent1, parent2 = population[idx1], population[idx2]
                f1, f2 = fitness_values[idx1], fitness_values[idx2]

                # 计算自适应交叉概率
                Pc = self.adaptive_Pc(max(f1, f2), avg_fitness, best_fitness)

                # 交叉操作
                child1, child2 = self.crossover(parent1, parent2, Pc)

                # 计算自适应变异概率
                Pm1 = self.adaptive_Pm(f1, avg_fitness, best_fitness)
                Pm2 = self.adaptive_Pm(f2, avg_fitness, best_fitness)

                # 变异操作
                child1 = self.mutation(child1, Pm1)
                child2 = self.mutation(child2, Pm2)

                # 计算子代适应度
                f_child1 = self.fitness(child1)
                f_child2 = self.fitness(child2)

                # Boltzmann选择
                if self.boltzmann_acceptance(f_child1 - f1, T):
                    new_population.append(child1)
                else:
                    new_population.append(parent1)

                if self.boltzmann_acceptance(f_child2 - f2, T):
                    new_population.append(child2)
                else:
                    new_population.append(parent2)

            # 确保种群大小不变
            if len(new_population) > self.pop_size:
                new_population = new_population[:self.pop_size]
            elif len(new_population) < self.pop_size:
                # 随机生成新个体补足
                while len(new_population) < self.pop_size:
                    new_population.append(np.random.random(4))

            # 更新种群和适应度
            population = np.array(new_population)
            fitness_values = np.array([self.fitness(ind) for ind in population])

            # 温度衰减
            T = T * self.alpha * self.beta

        # 找到最佳个体
        best_idx = np.argmax(fitness_values)
        best_individual = population[best_idx]
        best_fitness = fitness_values[best_idx]

        return best_individual, best_fitness, best_fitness_history, avg_fitness_history

    def plot_results(self, best_fitness_history: List[float], avg_fitness_history: List[float]):
        """
        绘制优化结果

        参数:
        best_fitness_history: 最佳适应度历史
        avg_fitness_history: 平均适应度历史
        """
        plt.figure(figsize=(12, 6))

        # 绘制适应度变化曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(best_fitness_history) + 1), best_fitness_history, 'r-', label='最佳适应度')
        plt.plot(range(1, len(avg_fitness_history) + 1), avg_fitness_history, 'b--', label='平均适应度')
        plt.xlabel('代数')
        plt.ylabel('适应度值 (阻断时间/秒)')
        plt.title('AE-SAGA优化过程 - 适应度变化')
        plt.legend()
        plt.grid(True)

        # 绘制适应度差值曲线
        plt.subplot(1, 2, 2)
        diff = np.array(best_fitness_history) - np.array(avg_fitness_history)
        plt.plot(range(1, len(diff) + 1), diff, 'g-', label='最佳与平均适应度差值')
        plt.xlabel('代数')
        plt.ylabel('适应度差值')
        plt.title('最佳与平均适应度差值变化')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('aesaga_optimization.png', dpi=300)
        plt.close()


# 运行优化算法
if __name__ == "__main__":
    # 设置中文字体支持
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

    # 创建优化器实例
    optimizer = AESAGA(
        pop_size=30,
        elite_size=0.1,
        Pc1=0.9,
        Pc2=0.6,
        Pm1=0.2,
        Pm2=0.001,
        T0=100.0,
        T_final=0.1,
        alpha=0.95,
        beta=1.0,
        max_stagnation=10
    )

    # 运行优化
    best_individual, best_fitness, best_history, avg_history = optimizer.optimize(generations=50)

    # 绘制结果
    optimizer.plot_results(best_history, avg_history)

    # 将最佳个体转换为实际参数
    best_vi = best_individual[0] * (140 - 70) + 70
    best_theta = best_individual[1] * 360
    best_tdrop = best_individual[2] * 47
    best_texpl_diff = best_individual[3] * 15
    best_texpl = best_tdrop + best_texpl_diff

    print("\n最佳参数:")
    print(f"速度: {best_vi:.2f} m/s")
    print(f"角度: {best_theta:.2f}°")
    print(f"释放时间: {best_tdrop:.2f} s")
    print(f"起爆时间差: {best_texpl_diff:.2f} s")
    print(f"起爆时间: {best_texpl:.2f} s")
    print(f"阻断时间: {best_fitness:.6f} s")

    # 保存最佳结果到文件
    with open('aesaga_best_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"最佳适应度(阻断时间): {best_fitness:.6f} s\n")
        f.write(f"速度: {best_vi:.2f} m/s\n")
        f.write(f"角度: {best_theta:.2f}°\n")
        f.write(f"释放时间: {best_tdrop:.2f} s\n")
        f.write(f"起爆时间差: {best_texpl_diff:.2f} s\n")
        f.write(f"起爆时间: {best_texpl:.2f} s\n")

    print("\n优化完成！结果已保存到 'aesaga_best_result.txt'")
    print("优化过程图表已保存到 'aesaga_optimization.png'")