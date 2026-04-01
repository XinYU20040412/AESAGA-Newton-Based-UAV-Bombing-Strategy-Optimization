import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

# 兼容无图形界面的运行环境（如 CI 或远程服务器）。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AESAGA:
    """
    自适应精英模拟退火遗传混合算法（AE-SAGA）
    结合遗传算法全局搜索能力和模拟退火接受准则。
    """

    MISSILE_INIT = np.array(
        [[20000.0, 0.0, 2000.0], [19000.0, 600.0, 2100.0], [18000.0, -600.0, 1900.0]],
        dtype=float,
    )
    MISSILE_VEL = np.array(
        [[-298.51, -0.0, -29.85], [-291.27, -9.2, -32.19], [-282.26, 9.41, -29.79]],
        dtype=float,
    )
    DRONE_INIT = np.array(
        [[17800.0, 0.0, 1800.0], [12000.0, 1400.0, 1400.0], [6000.0, -3000.0, 700.0], [11000.0, 2000.0, 1800.0], [13000.0, -2000.0, 1300.0]],
        dtype=float,
    )
    TRUE_TARGET_CENTER = np.array([0.0, 200.0, 5.0], dtype=float)
    TRUE_TARGET_RADIUS = 7.0
    TRUE_TARGET_HEIGHT = 10.0
    FAKE_TARGET = np.array([0.0, 0.0, 0.0], dtype=float)
    GRAVITY = 9.8
    SMOKE_RADIUS = 10.0
    SMOKE_SINK_SPEED = 3.0
    SMOKE_VALID_DURATION = 20.0

    def __init__(
        self,
        pop_size: int = 100,
        elite_size: float = 0.1,
        Pc1: float = 0.9,
        Pc2: float = 0.6,
        Pm1: float = 0.2,
        Pm2: float = 0.001,
        T0: Optional[float] = None,
        T_final: float = 0.001,
        alpha: float = 0.95,
        beta: float = 1.0,
        max_stagnation: int = 10,
        target_missile_index: int = 0,
        cover_threshold: float = 10.0,
        pace: int = 600,
        time_horizon: float = 67.0,
        tdrop_max: float = 5.0,
        texpl_delay_max: float = 5.0,
        bootstrap_samples: int = 400,
        random_seed: Optional[int] = None,
    ):
        if pop_size <= 4:
            raise ValueError("pop_size 必须大于 4")
        if not (0.0 < elite_size < 1.0):
            raise ValueError("elite_size 必须在 (0,1) 区间")
        if not (0 <= target_missile_index < self.MISSILE_INIT.shape[0]):
            raise ValueError("target_missile_index 越界")
        if pace <= 0:
            raise ValueError("pace 必须为正")

        if random_seed is not None:
            np.random.seed(int(random_seed))

        self.pop_size = int(pop_size)
        self.elite_size = max(1, int(self.pop_size * float(elite_size)))

        self.Pc1 = float(Pc1)
        self.Pc2 = float(Pc2)
        self.Pm1 = float(Pm1)
        self.Pm2 = float(Pm2)

        self.T0 = T0
        self.T_final = float(T_final)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_stagnation = int(max_stagnation)

        self.target_missile_index = int(target_missile_index)
        self.cover_threshold = float(cover_threshold)
        self.pace = int(pace)
        self.time_horizon = float(time_horizon)

        self.tdrop_max = float(tdrop_max)
        self.texpl_delay_max = float(texpl_delay_max)
        self.bootstrap_samples = int(bootstrap_samples)

        self.speed_center = 105.0
        self.speed_span = 70.0
        self.angle_span_deg = 30.0

        self._theta_base = self._init_theta(0, 0)
        self._fitness_cache: Dict[Tuple[float, float, float, float], float] = {}

        self._sample_points = self._generate_target_samples(
            n_samples_per_plane=21,
            z_values=(0.0, self.TRUE_TARGET_HEIGHT),
        )
        self._fy_init = self.DRONE_INIT[0].copy()

        # 预计算时间网格与目标导弹轨迹，避免 fitness 中重复构造。
        self._time_grid = np.linspace(0.0, self.time_horizon, self.pace, endpoint=False, dtype=float)
        self._delta_t = self.time_horizon / self.pace
        target_init = self.MISSILE_INIT[self.target_missile_index]
        target_vel = self.MISSILE_VEL[self.target_missile_index]
        self._missile_states = target_init + np.outer(self._time_grid, target_vel)
        self._missile_alive = self._missile_states[:, 2] > 0.0

    @staticmethod
    def _clip_prob(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    def _init_theta(self, idex_FY: int, index_M_bomb: int) -> float:
        FY = [
            [17800, 0, 1800],
            [12000, 1400, 1400],
            [6000, -3000, 700],
            [11000, 2000, 1800],
            [13000, -2000, 1300],
        ]
        M_int = [[20000, 0, 2000], [19000, 600, 2100], [18000, -600, 1900]]

        Xfy = FY[idex_FY][0]
        Yfy = FY[idex_FY][1]
        XG = (Xfy + M_int[index_M_bomb][0] + 0) / 3
        YG = (Yfy + M_int[index_M_bomb][1] + 200) / 3

        xx = XG - Xfy
        yy = YG - Yfy
        denom = float(np.hypot(xx, yy))
        if denom <= 1e-12:
            return 0.0

        cos_val = float(np.clip(xx / denom, -1.0, 1.0))
        theta = float(np.arccos(cos_val))
        if yy < 0:
            theta = float(2 * np.pi - theta)
        return theta

    def _target_missile_state(self, t: float) -> Optional[np.ndarray]:
        missile = self.MISSILE_INIT[self.target_missile_index] + self.MISSILE_VEL[self.target_missile_index] * t
        if missile[2] <= 0:
            return None
        return missile

    def _decode_individual(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 0.0, 1.0)

        vi = np.array([self.speed_center + x[0] * self.speed_span - self.speed_span / 2.0], dtype=float)
        theta = np.array(
            [self._theta_base + (x[1] * self.angle_span_deg - self.angle_span_deg / 2.0) * np.pi / 180.0],
            dtype=float,
        )
        tdrop = np.array([[x[2] * self.tdrop_max]], dtype=float)
        texpl = np.array([[tdrop[0, 0] + x[3] * self.texpl_delay_max]], dtype=float)
        return vi, theta, tdrop, texpl

    def _decode_individual_scalars(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        vi, theta, tdrop, texpl = self._decode_individual(x)
        return float(vi[0]), float(theta[0]), float(tdrop[0, 0]), float(texpl[0, 0])

    @staticmethod
    def _point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        seg = seg_end - seg_start
        seg_norm_sq = float(np.dot(seg, seg))
        if seg_norm_sq <= 1e-12:
            return float(np.linalg.norm(point - seg_start))

        ratio = float(np.dot(point - seg_start, seg) / seg_norm_sq)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        projection = seg_start + ratio * seg
        return float(np.linalg.norm(point - projection))

    def _generate_target_samples(self, n_samples_per_plane: int, z_values: Tuple[float, float]) -> np.ndarray:
        if n_samples_per_plane <= 0:
            raise ValueError("n_samples_per_plane 必须为正")

        angles = np.linspace(0.0, 2.0 * np.pi, int(n_samples_per_plane), endpoint=False)
        points: List[List[float]] = []
        for z_val in z_values:
            for angle in angles:
                points.append(
                    [
                        self.TRUE_TARGET_CENTER[0] + self.TRUE_TARGET_RADIUS * float(np.cos(angle)),
                        self.TRUE_TARGET_CENTER[1] + self.TRUE_TARGET_RADIUS * float(np.sin(angle)),
                        float(z_val),
                    ]
                )
        return np.asarray(points, dtype=float)

    def _smoke_centers_at_times(
        self,
        time_grid: np.ndarray,
        speed: float,
        theta: float,
        tdrop: float,
        texpl: float,
    ) -> np.ndarray:
        centers = np.full((len(time_grid), 3), np.nan, dtype=float)
        if texpl <= tdrop:
            return centers

        velocity = np.array([speed * np.cos(theta), speed * np.sin(theta), 0.0], dtype=float)
        drop_pos = self._fy_init + velocity * tdrop

        free_fall_duration = float(max(0.0, texpl - tdrop))
        expl_pos = drop_pos + velocity * free_fall_duration
        expl_pos[2] = drop_pos[2] - 0.5 * self.GRAVITY * free_fall_duration**2

        if expl_pos[2] <= 0.0:
            return centers

        active = (time_grid > texpl) & (time_grid < texpl + self.SMOKE_VALID_DURATION)
        if not np.any(active):
            return centers

        centers[active, 0] = expl_pos[0]
        centers[active, 1] = expl_pos[1]
        centers[active, 2] = expl_pos[2] - self.SMOKE_SINK_SPEED * (time_grid[active] - texpl)
        centers[centers[:, 2] <= 0.0] = np.nan
        return centers

    def _drone_positions_at_times(self, time_grid: np.ndarray, speed: float, theta: float) -> np.ndarray:
        velocity = np.array([speed * np.cos(theta), speed * np.sin(theta), 0.0], dtype=float)
        return self._fy_init + np.outer(time_grid, velocity)

    def _bomb_positions_at_times(
        self,
        time_grid: np.ndarray,
        speed: float,
        theta: float,
        tdrop: float,
        texpl: float,
    ) -> np.ndarray:
        bomb = np.full((len(time_grid), 3), np.nan, dtype=float)
        if texpl <= tdrop:
            return bomb

        velocity = np.array([speed * np.cos(theta), speed * np.sin(theta), 0.0], dtype=float)
        drop_pos = self._fy_init + velocity * tdrop
        flight_mask = (time_grid >= tdrop) & (time_grid < texpl)
        if not np.any(flight_mask):
            return bomb

        tau = time_grid[flight_mask] - tdrop
        bomb[flight_mask, 0] = drop_pos[0] + velocity[0] * tau
        bomb[flight_mask, 1] = drop_pos[1] + velocity[1] * tau
        bomb[flight_mask, 2] = drop_pos[2] - 0.5 * self.GRAVITY * tau**2
        bomb[bomb[:, 2] <= 0.0] = np.nan
        return bomb

    def _single_smoke_blocks_target(self, missile: np.ndarray, smoke: np.ndarray) -> bool:
        if np.any(np.isnan(smoke)):
            return False

        coarse_distance = self._point_to_segment_distance(smoke, missile, self.TRUE_TARGET_CENTER)
        if coarse_distance > self.cover_threshold + self.TRUE_TARGET_RADIUS:
            return False

        samples = self._sample_points
        line_vec = samples - missile
        line_sq = np.sum(line_vec * line_vec, axis=1)
        line_sq = np.clip(line_sq, 1e-12, None)

        missile_to_smoke = smoke - missile
        ratio = np.sum(line_vec * missile_to_smoke, axis=1) / line_sq
        ratio = np.clip(ratio, 0.0, 1.0)
        projection = missile + line_vec * ratio[:, None]

        dist_line = np.linalg.norm(smoke - projection, axis=1)
        cond = dist_line <= self.cover_threshold
        if not np.any(cond):
            return False

        dot_missile = np.sum((smoke - missile) * line_vec, axis=1)
        dist_missile_smoke = float(np.linalg.norm(missile_to_smoke))
        cond &= (dot_missile >= 0.0) | (dist_missile_smoke <= self.cover_threshold)

        sample_to_smoke = smoke - samples
        sample_to_missile = missile - samples
        dot_sample = np.sum(sample_to_smoke * sample_to_missile, axis=1)
        dist_sample_smoke = np.linalg.norm(sample_to_smoke, axis=1)
        cond &= (dot_sample >= 0.0) | (dist_sample_smoke <= self.cover_threshold)

        return bool(np.all(cond))

    def _cover_flags_for_times(
        self,
        time_grid: np.ndarray,
        missile_states: np.ndarray,
        smoke_centers: np.ndarray,
    ) -> np.ndarray:
        flags = np.zeros(len(time_grid), dtype=bool)
        missile_alive = missile_states[:, 2] > 0.0
        smoke_active = ~np.isnan(smoke_centers[:, 0])
        candidates = np.where(missile_alive & smoke_active)[0]

        for idx in candidates:
            if self._single_smoke_blocks_target(missile_states[idx], smoke_centers[idx]):
                flags[idx] = True
        return flags

    def _simulate_engagement(self, x: np.ndarray, time_grid: np.ndarray) -> Dict[str, np.ndarray]:
        speed, theta, tdrop, texpl = self._decode_individual_scalars(x)
        missile_init = self.MISSILE_INIT[self.target_missile_index]
        missile_vel = self.MISSILE_VEL[self.target_missile_index]
        missile_states = missile_init + np.outer(time_grid, missile_vel)

        drone_states = self._drone_positions_at_times(time_grid, speed, theta)
        bomb_states = self._bomb_positions_at_times(time_grid, speed, theta, tdrop, texpl)
        smoke_centers = self._smoke_centers_at_times(time_grid, speed, theta, tdrop, texpl)
        cover_flags = self._cover_flags_for_times(time_grid, missile_states, smoke_centers)

        return {
            "time_grid": time_grid,
            "missile_states": missile_states,
            "drone_states": drone_states,
            "bomb_states": bomb_states,
            "smoke_centers": smoke_centers,
            "cover_flags": cover_flags,
        }

    @staticmethod
    def _build_cover_intervals(time_grid: np.ndarray, flags: np.ndarray) -> List[Tuple[float, float]]:
        intervals: List[Tuple[float, float]] = []
        if len(time_grid) == 0:
            return intervals

        dt = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 0.0
        start: Optional[float] = None
        for idx, flag in enumerate(flags):
            if flag and start is None:
                start = float(time_grid[idx])
            elif not flag and start is not None:
                intervals.append((start, float(time_grid[idx])))
                start = None

        if start is not None:
            intervals.append((start, float(time_grid[-1] + dt)))
        return intervals

    def fitness(self, x: np.ndarray) -> float:
        """
        计算个体适应度（阻断时间）。

        参数:
        x: 个体基因 [速度比例, 角度比例, 释放时间比例, 起爆时间差比例]

        返回:
        适应度值（阻断时间）
        """
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 0.0, 1.0)
        key = tuple(np.round(x, 6))
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        try:
            simulation = self._simulate_engagement(x, self._time_grid)
            t_block = float(np.sum(simulation["cover_flags"]) * self._delta_t)
            t_block = float(max(0.0, t_block))
        except Exception as exc:
            logger.debug("fitness 计算失败，按0处理: %s", exc)
            t_block = 0.0

        self._fitness_cache[key] = t_block
        return t_block

    def adaptive_Pc(self, f_prime: float, f_avg: float, f_max: float) -> float:
        """自适应交叉概率计算。"""
        low = min(self.Pc1, self.Pc2)
        high = max(self.Pc1, self.Pc2)
        if f_max == f_avg:
            return self._clip_prob(self.Pc1, low, high)

        if f_prime >= f_avg:
            pc = self.Pc1 - (self.Pc1 - self.Pc2) * (f_prime - f_avg) / (f_max - f_avg)
            return self._clip_prob(pc, low, high)
        return self._clip_prob(self.Pc1, low, high)

    def adaptive_Pm(self, f: float, f_avg: float, f_max: float) -> float:
        """自适应变异概率计算（高适应度个体变异率更低）。"""
        low = min(self.Pm1, self.Pm2)
        high = max(self.Pm1, self.Pm2)
        if f_max == f_avg:
            return self._clip_prob(self.Pm1, low, high)

        if f >= f_avg:
            pm = self.Pm2 + (self.Pm1 - self.Pm2) * (f_max - f) / (f_max - f_avg)
            return self._clip_prob(pm, low, high)
        return self._clip_prob(self.Pm1, low, high)

    def initialize_population(self) -> np.ndarray:
        """初始化种群。"""
        pop = np.zeros((self.pop_size, 4), dtype=float)
        for dim in range(4):
            bins = (np.arange(self.pop_size, dtype=float) + np.random.random(self.pop_size)) / self.pop_size
            np.random.shuffle(bins)
            pop[:, dim] = bins

        # 时间相关维度采用偏置采样，提升可行区域命中概率。
        pop[:, 2] = np.random.beta(a=2.0, b=5.0, size=self.pop_size)
        pop[:, 3] = np.random.beta(a=2.0, b=5.0, size=self.pop_size)
        return np.clip(pop, 0.0, 1.0)

    def _bootstrap_population_if_flat(
        self, population: np.ndarray, fitness_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """当适应度全零时执行可行域探测，避免曲线长期贴地。"""
        if float(np.max(fitness_values)) > 0.0:
            return population, fitness_values, False

        best_candidate = None
        best_fit = -np.inf
        for _ in range(max(1, self.bootstrap_samples)):
            cand = np.random.random(4)
            cand[2] = np.random.beta(1.5, 6.0)
            cand[3] = np.random.beta(1.5, 6.0)
            f = self.fitness(cand)
            if f > best_fit:
                best_fit = f
                best_candidate = cand.copy()
            if best_fit > 0.0:
                break

        if best_candidate is not None and best_fit >= float(np.min(fitness_values)):
            worst_idx = int(np.argmin(fitness_values))
            population[worst_idx] = best_candidate
            fitness_values[worst_idx] = best_fit

        return population, fitness_values, bool(best_fit > 0.0)

    def selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """轮盘赌选择。"""
        adjusted = np.asarray(fitness_values, dtype=float).copy()
        min_fitness = np.min(adjusted)
        if min_fitness < 0:
            adjusted = adjusted - min_fitness + 1e-6

        fitness_sum = np.sum(adjusted)
        if fitness_sum == 0:
            selection_probs = np.ones(len(adjusted)) / len(adjusted)
        else:
            selection_probs = adjusted / fitness_sum

        selected_indices = np.random.choice(
            len(population),
            size=max(2, len(population) - self.elite_size),
            p=selection_probs,
            replace=True,
        )
        return selected_indices

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, Pc: float) -> Tuple[np.ndarray, np.ndarray]:
        """单点交叉。"""
        if np.random.random() > Pc:
            return parent1.copy(), parent2.copy()

        crossover_point = np.random.randint(1, 4)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutation(self, individual: np.ndarray, Pm: float) -> np.ndarray:
        """随机变异。"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < Pm:
                mutated[i] = np.random.random()
        return np.clip(mutated, 0.0, 1.0)

    def boltzmann_acceptance(self, delta_f: float, T: float) -> bool:
        """Boltzmann 接受准则。"""
        if delta_f >= 0:
            return True
        if T <= 1e-12:
            return False
        accept_prob = float(np.exp(np.clip(delta_f / T, -60.0, 0.0)))
        return bool(np.random.random() < accept_prob)

    def optimize(self, generations: int = 100) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """
        执行优化。

        参数:
        generations: 迭代代数

        返回:
        最佳个体, 最佳适应度, 最佳适应度历史, 平均适应度历史
        """
        self._fitness_cache.clear()

        population = self.initialize_population()
        fitness_values = np.array([self.fitness(ind) for ind in population], dtype=float)

        population, fitness_values, found_nonzero = self._bootstrap_population_if_flat(population, fitness_values)
        if not found_nonzero:
            logger.warning("初始种群适应度全零，已执行可行域探测，后续将继续自适应重采样。")

        if self.T0 is None:
            span = float(np.max(fitness_values) - np.min(fitness_values))
            self.T0 = max(1.0, span * 4.0)
        T = float(self.T0)

        best_fitness_history: List[float] = []
        avg_fitness_history: List[float] = []

        stagnation_count = 0
        prev_best_fitness = -np.inf

        for generation in range(int(generations)):
            best_idx = int(np.argmax(fitness_values))
            best_fitness = float(fitness_values[best_idx])
            avg_fitness = float(np.mean(fitness_values))

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            if best_fitness <= prev_best_fitness + 1e-12:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best_fitness = best_fitness

            if generation > 3 and stagnation_count >= 3:
                recent = np.array(best_fitness_history[-4:], dtype=float)
                baseline = max(abs(recent[0]), 1e-6)
                improvement_rate = (recent[-1] - recent[0]) / baseline
                if improvement_rate < 0.01:
                    self.beta = 0.9
                elif improvement_rate > 0.05:
                    self.beta = 1.0
                else:
                    self.beta = 0.95

            logger.info(
                "第%d代 - 最佳适应度: %.6f, 平均适应度: %.6f, 温度: %.6f",
                generation + 1,
                best_fitness,
                avg_fitness,
                T,
            )

            if T < self.T_final:
                logger.info("温度低于阈值，优化终止于第%d代", generation + 1)
                break

            if stagnation_count >= self.max_stagnation:
                if best_fitness <= 0.0 and generation < generations - 1:
                    logger.warning("连续停滞且最优仍为0，执行一次全量重采样。")
                    population = self.initialize_population()
                    fitness_values = np.array([self.fitness(ind) for ind in population], dtype=float)
                    population, fitness_values, _ = self._bootstrap_population_if_flat(population, fitness_values)
                    stagnation_count = 0
                    T = max(T, 0.5 * float(self.T0))
                    continue

                logger.info("达到最大停滞代数，优化终止于第%d代", generation + 1)
                break

            new_population: List[np.ndarray] = []

            elite_indices = np.argsort(fitness_values)[-self.elite_size :]
            elite_pool = population[elite_indices].tolist()
            new_population.extend(elite_pool)

            selected_indices = self.selection(population, fitness_values)
            for i in range(0, len(selected_indices), 2):
                if i + 1 >= len(selected_indices):
                    idx = int(selected_indices[i])
                    new_population.append(population[idx].copy())
                    break

                idx1 = int(selected_indices[i])
                idx2 = int(selected_indices[i + 1])
                parent1, parent2 = population[idx1], population[idx2]
                f1, f2 = float(fitness_values[idx1]), float(fitness_values[idx2])

                Pc = self.adaptive_Pc(max(f1, f2), avg_fitness, best_fitness)
                child1, child2 = self.crossover(parent1, parent2, Pc)

                Pm_a = self.adaptive_Pm(f1, avg_fitness, best_fitness)
                Pm_b = self.adaptive_Pm(f2, avg_fitness, best_fitness)
                child1 = self.mutation(child1, Pm_a)
                child2 = self.mutation(child2, Pm_b)

                f_child1 = self.fitness(child1)
                f_child2 = self.fitness(child2)

                if self.boltzmann_acceptance(f_child1 - f1, T):
                    new_population.append(child1)
                else:
                    new_population.append(parent1)

                if self.boltzmann_acceptance(f_child2 - f2, T):
                    new_population.append(child2)
                else:
                    new_population.append(parent2)

            if len(new_population) > self.pop_size:
                new_population = new_population[: self.pop_size]
            elif len(new_population) < self.pop_size:
                while len(new_population) < self.pop_size:
                    cand = np.random.random(4)
                    cand[2] = np.random.beta(2.0, 5.0)
                    cand[3] = np.random.beta(2.0, 5.0)
                    new_population.append(cand)

            population = np.asarray(new_population, dtype=float)
            fitness_values = np.array([self.fitness(ind) for ind in population], dtype=float)

            T = max(1e-8, T * self.alpha * self.beta)

        best_idx = int(np.argmax(fitness_values))
        best_individual = population[best_idx].copy()
        best_fitness = float(fitness_values[best_idx])
        return best_individual, best_fitness, best_fitness_history, avg_fitness_history

    def plot_results(
        self,
        best_fitness_history: List[float],
        avg_fitness_history: List[float],
        save_path: str,
    ) -> str:
        """绘制优化结果曲线并保存。"""
        if len(best_fitness_history) == 0:
            logger.warning("历史记录为空，跳过绘图。")
            return ""

        x = np.arange(1, len(best_fitness_history) + 1)
        best_arr = np.asarray(best_fitness_history, dtype=float)
        avg_arr = np.asarray(avg_fitness_history, dtype=float)

        fig, ax = plt.subplots(figsize=(12.8, 6.6), dpi=220)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        ax.plot(x, best_arr, color="#2563eb", linewidth=2.4, label="最优适应度")
        ax.plot(x, avg_arr, color="#f97316", linewidth=2.0, alpha=0.95, label="平均适应度")
        ax.fill_between(x, avg_arr, best_arr, color="#93c5fd", alpha=0.18)

        ax.set_xlabel("迭代次数", fontsize=12)
        ax.set_ylabel("适应度（阻断时间/s）", fontsize=12)
        ax.set_title("第二问 AESAGA 优化过程", fontsize=15, pad=10)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.legend(frameon=True)

        max_idx = int(np.argmax(best_arr))
        ax.scatter([x[max_idx]], [best_arr[max_idx]], color="#dc2626", s=42, zorder=5)
        ax.annotate(
            f"best={best_arr[max_idx]:.4f}s",
            xy=(x[max_idx], best_arr[max_idx]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=10,
            color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fee2e2", ec="#fecaca", alpha=0.95),
        )

        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def _save_workflow_diagram(self, save_path: str) -> str:
        """导出算法流程图，便于 README 一图说明思路。"""
        steps = [
            "初始化参数与编码边界",
            "生成种群并计算适应度",
            "精英保留 + 轮盘赌选择",
            "自适应交叉与变异",
            "Boltzmann 接受准则",
            "温度退火与停滞判定",
            "输出最优参数与可视化产物",
        ]

        fig, ax = plt.subplots(figsize=(8.0, 11.0), dpi=220)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        ax.axis("off")

        ys = np.linspace(0.92, 0.10, len(steps))
        for i, (y, text) in enumerate(zip(ys, steps)):
            ax.text(
                0.5,
                y,
                text,
                ha="center",
                va="center",
                fontsize=12,
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.45", fc="#dbeafe", ec="#60a5fa", lw=1.2),
            )
            if i < len(steps) - 1:
                ax.annotate(
                    "",
                    xy=(0.5, ys[i + 1] + 0.04),
                    xytext=(0.5, y - 0.04),
                    arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2),
                )

        ax.text(0.5, 0.97, "AESAGA 第二问求解流程", ha="center", va="center", fontsize=15, color="#1e3a8a")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def _save_history_gif(
        self,
        best_fitness_history: List[float],
        avg_fitness_history: List[float],
        save_path: str,
        fps: int = 12,
    ) -> str:
        """导出优化过程动态 GIF。"""
        if len(best_fitness_history) == 0:
            return ""

        x = np.arange(1, len(best_fitness_history) + 1)
        best_arr = np.asarray(best_fitness_history, dtype=float)
        avg_arr = np.asarray(avg_fitness_history, dtype=float)

        fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=150)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")

        def _update(frame: int):
            ax.clear()
            ax.set_facecolor("#f8fafc")
            n = frame + 1
            ax.plot(x[:n], best_arr[:n], color="#2563eb", linewidth=2.4, label="最优适应度")
            ax.plot(x[:n], avg_arr[:n], color="#f97316", linewidth=1.9, label="平均适应度")
            ax.set_xlim(1, len(x))
            y_min = float(np.min([best_arr.min(), avg_arr.min()]))
            y_max = float(np.max([best_arr.max(), avg_arr.max()]))
            pad = max(0.05 * (y_max - y_min), 0.05)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlabel("迭代次数")
            ax.set_ylabel("适应度（阻断时间/s）")
            ax.set_title("AESAGA 优化收敛动画")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="lower right")

        ani = animation.FuncAnimation(fig, _update, frames=len(x), interval=120, blit=False)
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, int(fps))))
        except Exception as exc:
            logger.warning("导出 GIF 失败: %s", exc)
            plt.close(fig)
            return ""

        plt.close(fig)
        return save_path

    def _draw_true_target_cylinder(self, ax) -> None:
        theta = np.linspace(0.0, 2.0 * np.pi, 36)
        z = np.linspace(0.0, self.TRUE_TARGET_HEIGHT, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x_grid = self.TRUE_TARGET_CENTER[0] + self.TRUE_TARGET_RADIUS * np.cos(theta_grid)
        y_grid = self.TRUE_TARGET_CENTER[1] + self.TRUE_TARGET_RADIUS * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color="#22c55e", alpha=0.14, linewidth=0, shade=False)

        x_circle = self.TRUE_TARGET_CENTER[0] + self.TRUE_TARGET_RADIUS * np.cos(theta)
        y_circle = self.TRUE_TARGET_CENTER[1] + self.TRUE_TARGET_RADIUS * np.sin(theta)
        ax.plot(x_circle, y_circle, np.zeros_like(theta), color="#22c55e", linewidth=1.1, alpha=0.7)
        ax.plot(
            x_circle,
            y_circle,
            np.full_like(theta, self.TRUE_TARGET_HEIGHT),
            color="#22c55e",
            linewidth=1.1,
            alpha=0.7,
        )

    @staticmethod
    def _compute_scene_limits(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        spans = np.maximum(maxs - mins, np.array([1.0, 1.0, 1.0], dtype=float))
        margins = np.maximum(0.08 * spans, np.array([300.0, 220.0, 120.0], dtype=float))
        return (
            (float(mins[0] - margins[0]), float(maxs[0] + margins[0])),
            (float(mins[1] - margins[1]), float(maxs[1] + margins[1])),
            (float(max(0.0, mins[2] - margins[2])), float(maxs[2] + margins[2])),
        )

    def _render_full_process_frame(
        self,
        ax3d,
        ax_timeline,
        simulation: Dict[str, np.ndarray],
        frame_idx: int,
        speed: float,
        tdrop: float,
        texpl: float,
        scene_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        sphere_unit: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        time_grid = simulation["time_grid"]
        missile_states = simulation["missile_states"]
        drone_states = simulation["drone_states"]
        bomb_states = simulation["bomb_states"]
        smoke_centers = simulation["smoke_centers"]
        cover_flags = simulation["cover_flags"]

        frame_idx = int(np.clip(frame_idx, 0, len(time_grid) - 1))
        now_t = float(time_grid[frame_idx])

        ax3d.clear()
        ax_timeline.clear()

        ax3d.set_facecolor("#0b1020")
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.set_pane_color((11 / 255, 16 / 255, 32 / 255, 1.0))

        ax3d.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

        missile_path = missile_states[:, :]
        drone_path = drone_states[:, :]
        ax3d.plot(missile_path[:, 0], missile_path[:, 1], missile_path[:, 2], color="#334155", linewidth=1.2, alpha=0.3)
        ax3d.plot(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], color="#334155", linewidth=1.2, alpha=0.3)

        ax3d.plot(
            missile_path[: frame_idx + 1, 0],
            missile_path[: frame_idx + 1, 1],
            missile_path[: frame_idx + 1, 2],
            color="#60a5fa",
            linewidth=2.1,
            label="导弹轨迹",
        )
        ax3d.plot(
            drone_path[: frame_idx + 1, 0],
            drone_path[: frame_idx + 1, 1],
            drone_path[: frame_idx + 1, 2],
            color="#f59e0b",
            linewidth=2.0,
            label="无人机轨迹",
        )

        current_missile = missile_states[frame_idx]
        current_drone = drone_states[frame_idx]
        ax3d.scatter(
            [current_missile[0]],
            [current_missile[1]],
            [current_missile[2]],
            color="#3b82f6",
            s=30,
            depthshade=False,
        )
        ax3d.scatter(
            [current_drone[0]],
            [current_drone[1]],
            [current_drone[2]],
            color="#f97316",
            s=30,
            depthshade=False,
        )

        bomb_path = bomb_states[: frame_idx + 1]
        valid_bomb_path = ~np.isnan(bomb_path[:, 0])
        if np.any(valid_bomb_path):
            ax3d.plot(
                bomb_path[valid_bomb_path, 0],
                bomb_path[valid_bomb_path, 1],
                bomb_path[valid_bomb_path, 2],
                color="#eab308",
                linewidth=1.9,
                alpha=0.95,
                label="弹道段",
            )
            bomb_now = bomb_states[frame_idx]
            if not np.isnan(bomb_now[0]):
                ax3d.scatter([bomb_now[0]], [bomb_now[1]], [bomb_now[2]], color="#fde047", s=26, depthshade=False)

        smoke_now = smoke_centers[frame_idx]
        if not np.isnan(smoke_now[0]):
            ux, uy, uz = sphere_unit
            sphere_x = smoke_now[0] + self.SMOKE_RADIUS * ux
            sphere_y = smoke_now[1] + self.SMOKE_RADIUS * uy
            sphere_z = smoke_now[2] + self.SMOKE_RADIUS * uz
            ax3d.plot_surface(
                sphere_x,
                sphere_y,
                sphere_z,
                color="#38bdf8",
                alpha=0.18,
                linewidth=0,
                shade=False,
            )
            ax3d.plot_wireframe(
                sphere_x,
                sphere_y,
                sphere_z,
                color="#7dd3fc",
                rstride=2,
                cstride=2,
                linewidth=0.3,
                alpha=0.35,
            )
            ax3d.scatter([smoke_now[0]], [smoke_now[1]], [smoke_now[2]], color="#67e8f9", s=18, depthshade=False)

        line_color = "#22c55e" if bool(cover_flags[frame_idx]) else "#f97316"
        ax3d.plot(
            [current_missile[0], self.TRUE_TARGET_CENTER[0]],
            [current_missile[1], self.TRUE_TARGET_CENTER[1]],
            [current_missile[2], self.TRUE_TARGET_CENTER[2]],
            color=line_color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.95,
        )

        ax3d.scatter(
            [self.FAKE_TARGET[0]],
            [self.FAKE_TARGET[1]],
            [self.FAKE_TARGET[2]],
            color="#f43f5e",
            marker="X",
            s=90,
            depthshade=False,
            label="假目标",
        )
        self._draw_true_target_cylinder(ax3d)

        ax3d.set_xlim(*scene_limits[0])
        ax3d.set_ylim(*scene_limits[1])
        ax3d.set_zlim(*scene_limits[2])
        ax3d.set_box_aspect((scene_limits[0][1] - scene_limits[0][0], 0.35 * (scene_limits[1][1] - scene_limits[1][0]), 0.25 * (scene_limits[2][1] - scene_limits[2][0])))
        ax3d.view_init(elev=22, azim=-62 + 0.10 * frame_idx)

        ax3d.set_xlabel("X (m)", color="#cbd5e1", labelpad=8)
        ax3d.set_ylabel("Y (m)", color="#cbd5e1", labelpad=8)
        ax3d.set_zlabel("Z (m)", color="#cbd5e1", labelpad=6)
        ax3d.tick_params(colors="#cbd5e1")
        ax3d.set_title(f"无人机投弹-爆炸-遮蔽 3D 流程 | t={now_t:.1f}s", color="#e2e8f0", fontsize=12, pad=12)
        ax3d.legend(loc="upper right", fontsize=8, frameon=False, labelcolor="#e2e8f0")

        status = cover_flags.astype(float)
        intervals = self._build_cover_intervals(time_grid, cover_flags)
        for start, end in intervals:
            ax_timeline.axvspan(start, end, color="#22c55e", alpha=0.12)

        ax_timeline.step(time_grid, status, where="post", color="#22c55e", linewidth=2.0, label="遮蔽状态")
        ax_timeline.axvline(now_t, color="#f8fafc", linestyle="--", linewidth=1.3, alpha=0.85)
        ax_timeline.axvline(tdrop, color="#f59e0b", linestyle=":", linewidth=1.2, alpha=0.9, label="投放时刻")
        ax_timeline.axvline(texpl, color="#38bdf8", linestyle=":", linewidth=1.2, alpha=0.9, label="起爆时刻")

        dt = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 0.0
        covered_now = float(np.sum(cover_flags[: frame_idx + 1]) * dt)
        total_cover = float(np.sum(cover_flags) * dt)

        ax_timeline.set_facecolor("#10172a")
        ax_timeline.set_ylim(-0.05, 1.15)
        ax_timeline.set_yticks([0, 1])
        ax_timeline.set_yticklabels(["未遮蔽", "有效遮蔽"], color="#cbd5e1")
        ax_timeline.tick_params(axis="x", colors="#cbd5e1")
        ax_timeline.grid(True, linestyle="--", alpha=0.2)
        ax_timeline.set_xlabel("时间 (s)", color="#cbd5e1")
        ax_timeline.set_title("遮蔽时间线", color="#e2e8f0", fontsize=11)
        ax_timeline.legend(loc="upper right", fontsize=8, frameon=False, labelcolor="#e2e8f0")
        ax_timeline.text(
            0.02,
            0.93,
            f"当前机速: {speed:.2f} m/s\n累计遮蔽: {covered_now:.2f} s\n总遮蔽: {total_cover:.2f} s",
            transform=ax_timeline.transAxes,
            color="#e2e8f0",
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1e293b", ec="#334155", alpha=0.92),
        )

    def _save_cover_timeline_figure(
        self,
        simulation: Dict[str, np.ndarray],
        tdrop: float,
        texpl: float,
        save_path: str,
    ) -> str:
        time_grid = simulation["time_grid"]
        cover_flags = simulation["cover_flags"]
        intervals = self._build_cover_intervals(time_grid, cover_flags)

        fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=220)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")

        for start, end in intervals:
            ax.axvspan(start, end, color="#86efac", alpha=0.34)

        ax.step(time_grid, cover_flags.astype(float), where="post", color="#16a34a", linewidth=2.2)
        ax.axvline(tdrop, color="#f59e0b", linestyle=":", linewidth=1.5, label="投放时刻")
        ax.axvline(texpl, color="#0284c7", linestyle=":", linewidth=1.5, label="起爆时刻")

        dt = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 0.0
        total_cover = float(np.sum(cover_flags) * dt)

        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["未遮蔽", "有效遮蔽"])
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("状态")
        ax.set_title(f"全流程遮蔽时间线（总有效遮蔽: {total_cover:.3f} s）")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right")

        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def _save_full_process_3d_gif(
        self,
        simulation: Dict[str, np.ndarray],
        speed: float,
        tdrop: float,
        texpl: float,
        save_path: str,
        fps: int = 14,
    ) -> str:
        time_grid = simulation["time_grid"]
        if len(time_grid) == 0:
            return ""

        missile_states = simulation["missile_states"]
        drone_states = simulation["drone_states"]
        smoke_centers = simulation["smoke_centers"]
        bomb_states = simulation["bomb_states"]

        points_for_limits = [missile_states, drone_states, np.array([self.TRUE_TARGET_CENTER, self.FAKE_TARGET], dtype=float)]
        valid_smoke = smoke_centers[~np.isnan(smoke_centers[:, 0])]
        if len(valid_smoke) > 0:
            points_for_limits.append(valid_smoke)
        valid_bomb = bomb_states[~np.isnan(bomb_states[:, 0])]
        if len(valid_bomb) > 0:
            points_for_limits.append(valid_bomb)
        all_points = np.vstack(points_for_limits)
        scene_limits = self._compute_scene_limits(all_points)

        u = np.linspace(0.0, 2.0 * np.pi, 16)
        v = np.linspace(0.0, np.pi, 11)
        sphere_unit = (
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        )

        fig = plt.figure(figsize=(14.0, 6.6), dpi=140)
        fig.patch.set_facecolor("#0b1020")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.08)
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax_timeline = fig.add_subplot(gs[0, 1])

        def _update(frame: int):
            self._render_full_process_frame(
                ax3d=ax3d,
                ax_timeline=ax_timeline,
                simulation=simulation,
                frame_idx=frame,
                speed=speed,
                tdrop=tdrop,
                texpl=texpl,
                scene_limits=scene_limits,
                sphere_unit=sphere_unit,
            )

        ani = animation.FuncAnimation(fig, _update, frames=len(time_grid), interval=max(1, int(1000 / max(1, int(fps)))), blit=False)
        try:
            ani.save(save_path, writer=animation.PillowWriter(fps=max(1, int(fps))))
        except Exception as exc:
            logger.warning("导出 3D 全流程 GIF 失败: %s", exc)
            plt.close(fig)
            return ""

        plt.close(fig)
        return save_path

    def _save_full_process_snapshot(
        self,
        simulation: Dict[str, np.ndarray],
        speed: float,
        tdrop: float,
        texpl: float,
        save_path: str,
    ) -> str:
        time_grid = simulation["time_grid"]
        cover_flags = simulation["cover_flags"]
        if len(time_grid) == 0:
            return ""

        if np.any(cover_flags):
            frame_idx = int(np.argmax(cover_flags))
        else:
            frame_idx = int(np.clip(np.searchsorted(time_grid, texpl + 0.6), 0, len(time_grid) - 1))

        missile_states = simulation["missile_states"]
        drone_states = simulation["drone_states"]
        smoke_centers = simulation["smoke_centers"]
        bomb_states = simulation["bomb_states"]

        points_for_limits = [missile_states, drone_states, np.array([self.TRUE_TARGET_CENTER, self.FAKE_TARGET], dtype=float)]
        valid_smoke = smoke_centers[~np.isnan(smoke_centers[:, 0])]
        if len(valid_smoke) > 0:
            points_for_limits.append(valid_smoke)
        valid_bomb = bomb_states[~np.isnan(bomb_states[:, 0])]
        if len(valid_bomb) > 0:
            points_for_limits.append(valid_bomb)
        scene_limits = self._compute_scene_limits(np.vstack(points_for_limits))

        u = np.linspace(0.0, 2.0 * np.pi, 16)
        v = np.linspace(0.0, np.pi, 11)
        sphere_unit = (
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        )

        fig = plt.figure(figsize=(14.0, 6.6), dpi=180)
        fig.patch.set_facecolor("#0b1020")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.08)
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax_timeline = fig.add_subplot(gs[0, 1])

        self._render_full_process_frame(
            ax3d=ax3d,
            ax_timeline=ax_timeline,
            simulation=simulation,
            frame_idx=frame_idx,
            speed=speed,
            tdrop=tdrop,
            texpl=texpl,
            scene_limits=scene_limits,
            sphere_unit=sphere_unit,
        )

        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=260, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def export_full_process_showcase(
        self,
        best_individual: np.ndarray,
        output_dir: str,
        frames: int = 120,
        fps: int = 14,
    ) -> Dict[str, object]:
        os.makedirs(output_dir, exist_ok=True)

        speed, theta, tdrop, texpl = self._decode_individual_scalars(best_individual)
        dense_sim = self._simulate_engagement(best_individual, self._time_grid)

        timeline_path = os.path.join(output_dir, "full_process_timeline.png")
        cover_png_path = os.path.join(output_dir, "full_process_3d.png")
        cover_gif_path = os.path.join(output_dir, "full_process_3d.gif")
        summary_path = os.path.join(output_dir, "full_process_summary.json")

        self._save_cover_timeline_figure(dense_sim, tdrop=tdrop, texpl=texpl, save_path=timeline_path)

        frame_count = max(48, int(frames))
        anim_time_grid = np.linspace(0.0, self.time_horizon, frame_count, endpoint=True, dtype=float)
        anim_sim = self._simulate_engagement(best_individual, anim_time_grid)

        self._save_full_process_snapshot(anim_sim, speed=speed, tdrop=tdrop, texpl=texpl, save_path=cover_png_path)
        gif_saved = self._save_full_process_3d_gif(
            anim_sim,
            speed=speed,
            tdrop=tdrop,
            texpl=texpl,
            save_path=cover_gif_path,
            fps=max(1, int(fps)),
        )

        intervals = self._build_cover_intervals(dense_sim["time_grid"], dense_sim["cover_flags"])
        total_cover = float(np.sum(dense_sim["cover_flags"]) * self._delta_t)
        summary = {
            "best_parameters": {
                "speed_mps": float(speed),
                "theta_deg": float(np.degrees(theta)),
                "tdrop_s": float(tdrop),
                "texpl_s": float(texpl),
                "texpl_delay_s": float(texpl - tdrop),
            },
            "cover_result": {
                "effective_cover_s": float(total_cover),
                "cover_intervals_s": [[float(start), float(end)] for start, end in intervals],
            },
            "visual_assets": {
                "hero_png": os.path.basename(cover_png_path),
                "hero_gif": os.path.basename(gif_saved) if gif_saved else None,
                "timeline_png": os.path.basename(timeline_path),
            },
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary

    def export_showcase(
        self,
        best_individual: np.ndarray,
        best_fitness: float,
        best_fitness_history: List[float],
        avg_fitness_history: List[float],
        output_dir: str,
        save_gif: bool = True,
        gif_fps: int = 12,
    ) -> Dict[str, object]:
        """导出 README 展示所需的全部产物。"""
        os.makedirs(output_dir, exist_ok=True)

        curve_path = os.path.join(output_dir, "aesaga_optimization.png")
        workflow_path = os.path.join(output_dir, "aesaga_workflow.png")
        gif_path = os.path.join(output_dir, "aesaga_optimization.gif")
        metrics_csv_path = os.path.join(output_dir, "aesaga_metrics.csv")
        summary_json_path = os.path.join(output_dir, "aesaga_summary.json")
        result_txt_path = os.path.join(output_dir, "aesaga_best_result.txt")

        self.plot_results(best_fitness_history, avg_fitness_history, curve_path)
        self._save_workflow_diagram(workflow_path)
        gif_saved = ""
        if save_gif:
            gif_saved = self._save_history_gif(best_fitness_history, avg_fitness_history, gif_path, fps=gif_fps)

        best_vi = self.speed_center + best_individual[0] * self.speed_span - self.speed_span / 2.0
        best_theta = np.degrees(
            self._theta_base + (best_individual[1] * self.angle_span_deg - self.angle_span_deg / 2.0) * np.pi / 180.0
        )
        best_tdrop = best_individual[2] * self.tdrop_max
        best_texpl_diff = best_individual[3] * self.texpl_delay_max
        best_texpl = best_tdrop + best_texpl_diff

        with open(metrics_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "avg_fitness"])
            for idx, (best_f, avg_f) in enumerate(zip(best_fitness_history, avg_fitness_history), start=1):
                writer.writerow([idx, f"{best_f:.8f}", f"{avg_f:.8f}"])

        with open(result_txt_path, "w", encoding="utf-8") as f:
            f.write(f"最佳适应度(阻断时间): {best_fitness:.6f} s\n")
            f.write(f"速度: {best_vi:.2f} m/s\n")
            f.write(f"角度: {best_theta:.2f}°\n")
            f.write(f"释放时间: {best_tdrop:.2f} s\n")
            f.write(f"起爆时间差: {best_texpl_diff:.2f} s\n")
            f.write(f"起爆时间: {best_texpl:.2f} s\n")

        summary = {
            "best_fitness": float(best_fitness),
            "best_parameters": {
                "speed_mps": float(best_vi),
                "theta_deg": float(best_theta),
                "tdrop_s": float(best_tdrop),
                "texpl_delay_s": float(best_texpl_diff),
                "texpl_s": float(best_texpl),
            },
            "history": {
                "generations": len(best_fitness_history),
                "best_max": float(np.max(best_fitness_history)) if best_fitness_history else 0.0,
                "best_mean": float(np.mean(best_fitness_history)) if best_fitness_history else 0.0,
                "avg_final": float(avg_fitness_history[-1]) if avg_fitness_history else 0.0,
            },
            "output_files": {
                "curve_png": os.path.basename(curve_path),
                "workflow_png": os.path.basename(workflow_path),
                "metrics_csv": os.path.basename(metrics_csv_path),
                "result_txt": os.path.basename(result_txt_path),
                "gif": os.path.basename(gif_saved) if gif_saved else None,
            },
        }

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第二问 AESAGA 优化与展示产物导出")
    parser.add_argument("--generations", type=int, default=200, help="优化迭代代数")
    parser.add_argument("--pop-size", type=int, default=30, help="种群规模")
    parser.add_argument("--pace", type=int, default=600, help="每次 fitness 评估的时间采样点数")
    parser.add_argument("--random-seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gif-fps", type=int, default=12, help="优化过程 GIF 帧率")
    parser.add_argument("--skip-gif", action="store_true", help="不导出 GIF")
    parser.add_argument("--skip-hero", action="store_true", help="不导出3D全流程封面素材")
    parser.add_argument("--hero-fps", type=int, default=14, help="3D全流程 GIF 帧率")
    parser.add_argument("--hero-frames", type=int, default=120, help="3D全流程动画总帧数")

    default_output = Path(__file__).resolve().parents[2] / "docs" / "showcase" / "second_question"
    default_hero_output = Path(__file__).resolve().parents[2] / "docs" / "showcase" / "full_process"
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output),
        help="展示产物输出目录",
    )
    parser.add_argument(
        "--hero-output-dir",
        type=str,
        default=str(default_hero_output),
        help="3D全流程展示产物输出目录",
    )
    return parser


def main() -> None:
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        logger.warning("无法设置中文字体，图表中的中文可能无法正确显示")

    args = _build_parser().parse_args()

    optimizer = AESAGA(
        pop_size=int(args.pop_size),
        elite_size=0.1,
        Pc1=0.9,
        Pc2=0.6,
        Pm1=0.2,
        Pm2=0.001,
        T0=100.0,
        T_final=0.1,
        alpha=0.95,
        beta=1.0,
        max_stagnation=12,
        target_missile_index=0,
        pace=int(args.pace),
        tdrop_max=5.0,
        texpl_delay_max=5.0,
        bootstrap_samples=400,
        random_seed=int(args.random_seed),
    )

    best_individual, best_fitness, best_history, avg_history = optimizer.optimize(generations=int(args.generations))
    summary = optimizer.export_showcase(
        best_individual=best_individual,
        best_fitness=best_fitness,
        best_fitness_history=best_history,
        avg_fitness_history=avg_history,
        output_dir=str(args.output_dir),
        save_gif=not bool(args.skip_gif),
        gif_fps=int(args.gif_fps),
    )

    hero_summary: Optional[Dict[str, object]] = None
    if not bool(args.skip_hero):
        hero_summary = optimizer.export_full_process_showcase(
            best_individual=best_individual,
            output_dir=str(args.hero_output_dir),
            frames=int(args.hero_frames),
            fps=int(args.hero_fps),
        )

    params = summary["best_parameters"]
    print("\n最佳参数:")
    print(f"速度: {params['speed_mps']:.2f} m/s")
    print(f"角度: {params['theta_deg']:.2f}°")
    print(f"释放时间: {params['tdrop_s']:.2f} s")
    print(f"起爆时间差: {params['texpl_delay_s']:.2f} s")
    print(f"起爆时间: {params['texpl_s']:.2f} s")
    print(f"阻断时间: {summary['best_fitness']:.6f} s")

    print("\n优化完成，展示产物已导出:")
    print(str(Path(args.output_dir).resolve()))
    if hero_summary is not None:
        print("\n3D 全流程封面产物已导出:")
        print(str(Path(args.hero_output_dir).resolve()))


if __name__ == "__main__":
    main()
