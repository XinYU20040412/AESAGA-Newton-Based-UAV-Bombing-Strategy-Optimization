import numpy as np
from typing import List


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """
    计算点到线段的距离
    """
    line_vec = line_end - line_start
    line_length = np.linalg.norm(line_vec)

    if line_length == 0:
        return np.linalg.norm(point - line_start)

    point_vec = point - line_start
    t = np.dot(point_vec, line_vec) / (line_length ** 2)
    t = max(0, min(1, t))

    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def calculate_angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量之间的夹角（弧度）
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if norm_product == 0:
        return 0

    # 防止浮点误差导致值超出[-1, 1]范围
    cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    return np.arccos(cos_angle)


def is_obtuse_angle(vec1: np.ndarray, vec2: np.ndarray) -> bool:
    """
    判断两个向量之间的夹角是否为钝角
    """
    dot_product = np.dot(vec1, vec2)
    return dot_product < 0

def generate_circle_samples(n: int, z_values: List[float] = [0, 10]) -> List[List[float]]:
    """
    生成多个z平面上的圆形样本点
    """
    if n <= 0:
        raise ValueError("样本点数量n必须大于0")

    center_x, center_y = 0, 200
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    samples = []
    for z in z_values:
        for angle in angles:
            x = center_x + 7 * np.cos(angle)
            y = center_y + 7 * np.sin(angle)
            samples.append([x, y, z])

    return samples

class AdvancedMissileSmokeChecker:
    """
    高级导弹-烟雾检查器，使用新的判断条件
    """

    def __init__(self, n_samples_per_plane: int = 21, z_values: List[float] = [0, 10]):
        self.sample_points = generate_circle_samples(n_samples_per_plane, z_values)
        self.sample_points_np = np.array(self.sample_points)
        print(f"初始化完成，共生成 {len(self.sample_points)} 个样本点")

    def check_single_pair(self, missile: np.ndarray, sample: np.ndarray, smoke: np.ndarray, threshold: float) -> bool:
        """
        检查单个missile-sample-smoke组合
        """
        # 1. 计算点到线段的距离
        distance_to_line = point_to_line_distance(smoke, missile, sample)

        if distance_to_line > threshold:
            return False

        # 计算各个向量
        missile_to_sample = sample - missile
        missile_to_smoke = smoke - missile
        sample_to_smoke = smoke - sample
        sample_to_missile = missile - sample

        # 2a. 检查第一个夹角条件
        if is_obtuse_angle(missile_to_smoke, missile_to_sample):
            distance_missile_smoke = np.linalg.norm(missile_to_smoke)
            if distance_missile_smoke > threshold:
                return False

        # 2b. 检查第二个夹角条件
        if is_obtuse_angle(sample_to_smoke, sample_to_missile):
            distance_sample_smoke = np.linalg.norm(sample_to_smoke)
            if distance_sample_smoke > threshold:
                return False

        return True

    def check(self, missiles: List[List[float]], smokes: List[List[float]], threshold: float = 10) -> bool:
        """
        主检查函数
        """
        missiles_np = np.array(missiles)
        smokes_np = np.array(smokes)

        if smokes_np.size == 0 or missiles_np.size == 0:
            return False


        for missile in missiles_np:
            for sample in self.sample_points_np:
                for i in range(len(smokes_np)):
                    if self.check_single_pair(missile, sample, smokes_np[i], threshold):
                        break
                    elif i == len(smokes_np)-1 and not self.check_single_pair(missile, sample, smokes_np[i], threshold):
                        return False

        return True


# 使用类版本
if __name__ == "__main__":
    checker = AdvancedMissileSmokeChecker(n_samples_per_plane=4, z_values=[0, 10])

    # 测试用例
    test_missiles = [[10, 20, 30]]
    test_smokes = np.array([])

    result = checker.check(test_missiles, test_smokes)
    print(f"高级检查器结果: {result}")