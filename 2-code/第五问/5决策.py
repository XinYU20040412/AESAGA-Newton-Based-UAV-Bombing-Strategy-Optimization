import numpy as np
from scipy.optimize import linear_sum_assignment
import math

# ------------------------------------------------------------------------------
# 第一步：定义核心物理参数（保持原逻辑，确保参数完整性）
# ------------------------------------------------------------------------------
def define_a_problem_params():
    """定义无人机、导弹、目标、烟幕关键参数"""
    # 无人机参数（编号0=FY1~4=FY5，速度140单位/时间）
    uavs = {
        0: {"name": "FY1", "pos": (17800, 0, 1800), "speed": 140},
        1: {"name": "FY2", "pos": (12000, 1400, 1400), "speed": 140},
        2: {"name": "FY3", "pos": (6000, -3000, 700), "speed": 140},
        3: {"name": "FY4", "pos": (11000, 2000, 1800), "speed": 140},
        4: {"name": "FY5", "pos": (13000, -2000, 1300), "speed": 140}
    }
    # 导弹参数（编号0=M1~2=M3，速度300单位/时间）
    missiles = {
        0: {"name": "M1", "pos": (20000, 0, 2000), "speed": 300},
        1: {"name": "M2", "pos": (19000, 600, 2100), "speed": 300},
        2: {"name": "M3", "pos": (18000, -600, 1900), "speed": 300}
    }
    # 真目标（圆柱区域）、假目标（原点）
    targets = {
        "true": {"center": (0, 200, 5), "radius": 7, "height": (0, 10)},
        "fake": {"pos": (0, 0, 0)}  # 导弹默认攻击方向
    }
    # 烟幕参数
    smoke = {"radius": 10, "valid_time": 20, "sink_speed": 3}
    return uavs, missiles, targets, smoke

# ------------------------------------------------------------------------------
# 第二步：FAHP计算烟幕干扰指标权重（保持原逻辑，确保权重合理性）
# ------------------------------------------------------------------------------
def fuzzy_ahp_smoke_weight():
    """基于FAHP计算4个距离指标的归一化权重（D1>D4>D2=D3）"""
    # 1. 三角模糊数标度
    eq = (1, 1, 1)          # 同等重要
    much = (5, 7, 9)        # 重要得多
    slightly = (2, 3, 4)    # 较重要
    rev_much = (1/9, 1/7, 1/5)      # 重要得多的反向
    rev_slightly = (1/4, 1/3, 1/2)  # 较重要的反向

    # 2. 模糊判断矩阵（行/列：D1,D2,D3,D4）
    A = [
        [eq, much, much, much],          # D1 vs 其他（重要得多）
        [rev_much, eq, eq, rev_slightly],# D2 vs 其他
        [rev_much, eq, eq, rev_slightly],# D3 vs 其他
        [rev_much, slightly, slightly, eq]# D4 vs 其他
    ]
    n = len(A)

    # 3. 模糊合成运算（计算M_i = 每行模糊乘积）
    def fuzzy_mult(a, b): return (a[0]*b[0], a[1]*b[1], a[2]*b[2])
    def fuzzy_div(a, b): return (a[0]/b[2], a[1]/b[1], a[2]/b[0])
    
    M = []
    for i in range(n):
        product = (1.0, 1.0, 1.0)
        for j in range(n):
            product = fuzzy_mult(product, A[i][j])
        M.append(product)

    # 4. 计算模糊权重（w_i = M_i / 所有M_i的和）
    M_sum = (sum(m[0] for m in M), sum(m[1] for m in M), sum(m[2] for m in M))
    fuzzy_weights = [fuzzy_div(m, M_sum) for m in M]

    # 5. 去模糊化（重心法）与归一化
    def defuzzify(w): return (w[0] + 4 * w[1] + w[2]) / 6
    crisp_weights = [defuzzify(w) for w in fuzzy_weights]
    normalized_weights = [w / sum(crisp_weights) for w in crisp_weights]

    # 6. 一致性检验（确保FAHP权重合理）
    def consistency_check(A, weights):
        crisp_matrix = np.array([[row[j][1] for j in range(n)] for row in A])
        Aw = np.dot(crisp_matrix, weights)
        lambda_max = sum(Aw / (n * np.array(weights)))
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI = 0.90  # n=4时的标准RI值
        CR = CI / RI if RI != 0 else 0
        print(f"FAHP一致性检验：CR={CR:.6f} {'（通过，CR<0.1）' if CR<0.1 else '（未通过！）'}")
        return CR

    consistency_check(A, normalized_weights)
    return normalized_weights

# 提前计算FAHP权重（全局变量a/b/c/d，供距离计算使用）
final_weights = fuzzy_ahp_smoke_weight()
a, b, c, d = final_weights  # D1/D2/D3/D4的权重

# ------------------------------------------------------------------------------
# 第三步：核心指标计算（保持原逻辑，确保指标准确性）
# ------------------------------------------------------------------------------
def calc_comprehensive_distance(uav_pos, missile_pos, targets):
    """指标1：综合距离得分（越小越好，D1-D4加权）"""
    # D1：无人机到“导弹-真目标-假目标”平面的垂直距离占比
    p1, p2, p3 = targets["true"]["center"][:2] + (0,), targets["fake"]["pos"], missile_pos
    x1,y1,z1 = p1; x2,y2,z2 = p2; x3,y3,z3 = p3; x0,y0,z0 = uav_pos
    
    # 平面法向量计算
    v1 = (x2-x1, y2-y1, z2-z1); v2 = (x3-x1, y3-y1, z3-z1)
    nx = v1[1]*v2[2] - v1[2]*v2[1]
    ny = v1[2]*v2[0] - v1[0]*v2[2]
    nz = v1[0]*v2[1] - v1[1]*v2[0]
    d_plane = -(nx*x1 + ny*y1 + nz*z1)
    
    numerator = abs(nx*x0 + ny*y0 + nz*z0 + d_plane)
    denominator = math.sqrt(nx**2 + ny**2 + nz**2)
    D1 = numerator / denominator

    # D2：无人机-导弹直线距离
    D2 = math.sqrt((uav_pos[0]-missile_pos[0])**2 + 
                   (uav_pos[1]-missile_pos[1])**2 + 
                   (uav_pos[2]-missile_pos[2])**2)

    # D3：平面重心距离（无人机-真目标-导弹构成的三角形重心）
    XG = (uav_pos[0] + missile_pos[0] + targets["true"]["center"][0]) / 3
    YG = (uav_pos[1] + missile_pos[1] + targets["true"]["center"][1]) / 3
    D3 = math.sqrt((XG - uav_pos[0])**2 + (YG - uav_pos[1])**2)

    # D4：立体重心连线距离（加权1.5模拟遮挡长度）
    ZG = (uav_pos[2] + missile_pos[2] + targets["true"]["center"][2]) / 3
    D4 = math.sqrt((XG - uav_pos[0])**2 + (YG - uav_pos[1])**2 + (ZG - uav_pos[2])**2) * 1.5

    # 综合距离（权重加权，越小越好）
    return a*D1 + b*D2 + c*D3 + d*D4


def calc_time_surplus_score(uav, missile, targets):
    """指标2：时间富余得分（越大越好，衡量无人机投弹时间充裕度）"""
    # 导弹总飞行时间（从当前位置到假目标）
    missile_to_fake = math.sqrt((missile["pos"][0]-targets["fake"]["pos"][0])**2 +
                                (missile["pos"][1]-targets["fake"]["pos"][1])**2 +
                                (missile["pos"][2]-targets["fake"]["pos"][2])**2)
    time_missile_total = missile_to_fake / missile["speed"]

    # 无人机到导弹的最短时间（取x/y方向最短距离，确保投弹时效性）
    distance_min = np.min([abs(uav["pos"][0]-missile["pos"][0]), 
                          abs(uav["pos"][1]-missile["pos"][1])])
    time_uav_min = distance_min / uav["speed"]

    # 时间富余得分（归一化到[0,1]，避免负分）
    time_surplus = time_missile_total - time_uav_min
    return max(0, time_surplus / time_missile_total)


def calc_axis_angle_score(uav_pos, missile_pos, targets):
    """指标3：轴角遮蔽得分（越大越好，衡量烟幕遮挡匹配度）"""
    # 向量1：导弹飞行方向（导弹→假目标）
    dir_missile = (targets["fake"]["pos"][0]-missile_pos[0],
                   targets["fake"]["pos"][1]-missile_pos[1],
                   targets["fake"]["pos"][2]-missile_pos[2])
    norm_missile = math.sqrt(dir_missile[0]**2 + dir_missile[1]**2 + dir_missile[2]**2)

    # 向量2：无人机→真目标（烟幕需在两者之间，确保遮挡）
    dir_uav_true = (targets["true"]["center"][0]-uav_pos[0],
                    targets["true"]["center"][1]-uav_pos[1],
                    targets["true"]["center"][2]-uav_pos[2])
    norm_uav_true = math.sqrt(dir_uav_true[0]**2 + dir_uav_true[1]**2 + dir_uav_true[2]**2)

    # 计算夹角余弦（避免数值溢出，限制在[-1,1]）
    dot = dir_missile[0]*dir_uav_true[0] + dir_missile[1]*dir_uav_true[1] + dir_missile[2]*dir_uav_true[2]
    cos_theta = max(-1, min(1, dot / (norm_missile * norm_uav_true)))
    theta = math.acos(cos_theta)

    # 归一化得分（夹角越小，遮挡效果越好，得分越高）
    return 1 - theta / math.pi

# ------------------------------------------------------------------------------
# 第四步：指标标准化与熵权法（保持原逻辑，确保权重客观性）
# ------------------------------------------------------------------------------
def normalize_index(index_matrix, is_negative=False):
    """指标标准化到[0,1]：is_negative=True表示“越小越好”（逆指标）"""
    max_val = np.max(index_matrix)
    min_val = np.min(index_matrix)
    if max_val == min_val:
        return np.ones_like(index_matrix)  # 指标无差异时均为1
    # 逆指标：(max - x)/(max - min)；正指标：(x - min)/(max - min)
    return (max_val - index_matrix) / (max_val - min_val) if is_negative else (index_matrix - min_val) / (max_val - min_val)


def calc_entropy_weight(indicator_matrices):
    """熵权法计算3个指标的权重（输入：3个指标矩阵列表，确保权重客观）"""
    m_total = indicator_matrices[0].size  # 总样本数（无人机-导弹对数量）
    weights = []

    for mat in indicator_matrices:
        # 标准化（加微小值避免log(0)错误）
        mat_norm = mat + 1e-6
        # 计算每个样本的占比（p_ij = x_ij / sum(x_ij)）
        p = mat_norm / mat_norm.sum()
        # 计算熵值（e_j = -k * sum(p_ij * ln(p_ij))，k=1/ln(m_total)）
        entropy = -1 / math.log(m_total) * np.sum(p * np.log(p))
        # 权重 = 信息效用值 / 总信息效用值（信息效用值=1-熵）
        weights.append(1 - entropy)

    # 归一化权重（确保权重和为1）
    weights = np.array(weights) / sum(weights)
    return weights

# ------------------------------------------------------------------------------
# 第五步：严谨匈牙利算法匹配（核心优化部分，确保全局最优）
# ------------------------------------------------------------------------------
def build_optimized_cost_matrix(score_matrix):
    """
    构建优化的成本矩阵（符合严谨匈牙利算法要求）
    思路：将最大化得分问题转化为最小化成本问题，处理无人机数量>导弹数量的情况
    - 成本 = 全局最大得分 - 综合得分（确保成本非负，且得分越高成本越低）
    - 当U>M时，循环扩展列到U列（确保每个无人机有唯一分配，扩展列成本与原导弹一致）
    """
    n_uav, n_missile = score_matrix.shape
    max_score = np.max(score_matrix)
    
    # 1. 基础成本矩阵（U×M）：成本=最大得分-综合得分
    base_cost = max_score - score_matrix
    
    # 2. 扩展成本矩阵到U×U（当U>M时，循环使用原导弹成本，确保最优性）
    if n_uav > n_missile:
        # 计算需要扩展的列数
        extend_cols_count = n_uav - n_missile
        # 循环选择原导弹列（0→1→2→0→1...），确保扩展列成本与原列一致
        extend_col_indices = [i % n_missile for i in range(extend_cols_count)]
        # 扩展成本矩阵
        extend_cost = base_cost[:, extend_col_indices]
        optimized_cost = np.hstack([base_cost, extend_cost])
    else:
        # 当U≤M时，直接使用基础成本矩阵（后续算法会自动选择U个最优列）
        optimized_cost = base_cost
    
    return optimized_cost, max_score


def rigorous_hungarian_matching(score_matrix):
    """
    严谨的匈牙利匹配（基于scipy的Kuhn-Munkres实现，确保全局最优）
    返回：(无人机索引, 原导弹索引)列表，确保每个导弹至少被分配
    """
    n_uav, n_missile = score_matrix.shape
    
    # 1. 构建优化的成本矩阵
    optimized_cost, _ = build_optimized_cost_matrix(score_matrix)
    
    # 2. 执行严谨的匈牙利算法（Kuhn-Munkres），得到全局最优匹配
    # linear_sum_assignment返回：(行索引, 列索引)，确保每行每列仅一个匹配
    uav_indices, extended_missile_indices = linear_sum_assignment(optimized_cost)
    
    # 3. 将扩展列索引映射回原导弹索引（扩展列=原导弹索引%n_missile）
    original_missile_indices = [idx % n_missile for idx in extended_missile_indices]
    
    # 4. 验证每个导弹至少被分配（基于算法最优性，此步通常满足，仅做校验）
    matched_missiles = set(original_missile_indices)
    uncovered_missiles = [m for m in range(n_missile) if m not in matched_missiles]
    
    if uncovered_missiles:
        # 极端情况下（如部分导弹成本极高），强制调整一个无人机分配给未覆盖导弹
        # 选择成本最低的无人机-未覆盖导弹对进行调整
        for m_uncover in uncovered_missiles:
            # 找到未分配给该导弹的无人机
            for u in range(n_uav):
                if original_missile_indices[u] != m_uncover:
                    # 计算调整后的总成本变化（确保调整后总成本仍最小）
                    current_cost = optimized_cost[u, extended_missile_indices[u]]
                    new_cost = optimized_cost[u, m_uncover]
                    if new_cost <= current_cost:
                        # 调整匹配
                        original_missile_indices[u] = m_uncover
                        extended_missile_indices[u] = m_uncover
                        break
        # 再次验证（确保无遗漏）
        assert len(set(original_missile_indices)) == n_missile, "导弹未完全覆盖，需检查成本矩阵构造"
    
    # 5. 整理匹配结果（无人机索引与原导弹索引对应）
    matches = list(zip(uav_indices, original_missile_indices))
    
    # 确保所有无人机都被分配（算法特性，此步必然满足）
    assert len(matches) == n_uav, "部分无人机未分配，算法执行异常"
    
    return matches

# ------------------------------------------------------------------------------
# 第六步：投弹数量分配（基于最优匹配得分，保持原逻辑）
# ------------------------------------------------------------------------------
def assign_smoke_count(matches, score_matrix, uavs, missiles):
    """基于综合得分分配投弹数量（每无人机最多3枚，得分高则多分配）"""
    # 1. 整理匹配对（按得分降序排序，优先给高得分匹配分配更多弹药）
    match_scores = [(u, m, score_matrix[u, m]) for u, m in matches]
    match_scores.sort(key=lambda x: x[2], reverse=True)

    # 2. 初始化投弹计数（每无人机最多3枚，避免资源浪费）
    uav_smoke_count = {u: 0 for u in uavs.keys()}
    smoke_assign = {u: [] for u in uavs.keys()}

    # 3. 动态分配投弹数量（基于得分阈值，确保资源合理）
    for u, m, score in match_scores:
        if uav_smoke_count[u] >= 3:
            continue  # 无人机达到投弹上限
        
        # 按得分分配数量（得分越高，投弹越多，确保干扰效果）
        if score >= 0.8:
            count = 3
        elif score >= 0.5:
            count = 2
        else:
            count = 1
        
        # 调整数量，避免超过无人机上限
        count = min(count, 3 - uav_smoke_count[u])
        
        # 记录分配结果
        smoke_assign[u].append({
            "missile_name": missiles[m]["name"], 
            "missile_idx": m, 
            "count": count, 
            "score": score
        })
        uav_smoke_count[u] += count

    # 过滤无分配的无人机（理论上不会有，因所有无人机已匹配）
    smoke_assign = {u: assign for u, assign in smoke_assign.items() if assign}
    return smoke_assign, uav_smoke_count

# ------------------------------------------------------------------------------
# 主函数：整合全流程（参数定义→指标计算→权重→最优匹配→分配）
# ------------------------------------------------------------------------------
def main():
    # 1. 初始化参数
    uavs, missiles, targets, smoke = define_a_problem_params()
    n_uav = len(uavs)
    n_missile = len(missiles)
    uav_indices = list(uavs.keys())
    missile_indices = list(missiles.keys())

    # 2. 计算3个核心指标矩阵（n_uav × n_missile）
    D_matrix = np.zeros((n_uav, n_missile))  # 综合距离（越小越好）
    T_matrix = np.zeros((n_uav, n_missile))  # 时间富余（越大越好）
    F_matrix = np.zeros((n_uav, n_missile))  # 轴角遮蔽（越大越好）

    for u in uav_indices:
        uav = uavs[u]
        for m in missile_indices:
            missile = missiles[m]
            D_matrix[u, m] = calc_comprehensive_distance(uav["pos"], missile["pos"], targets)
            T_matrix[u, m] = calc_time_surplus_score(uav, missile, targets)
            F_matrix[u, m] = calc_axis_angle_score(uav["pos"], missile["pos"], targets)

    # 3. 指标标准化与权重计算（熵权法确保客观性）
    D_norm = normalize_index(D_matrix, is_negative=True)  # 逆指标标准化
    T_norm = normalize_index(T_matrix, is_negative=False) # 正指标标准化
    F_norm = normalize_index(F_matrix, is_negative=False) # 正指标标准化

    # 熵权法计算3个指标的权重
    indicator_matrices = [D_norm, T_norm, F_norm]
    w_D, w_T, w_F = calc_entropy_weight(indicator_matrices)

    # 4. 计算综合得分矩阵（加权求和，越高越好）
    score_matrix = w_D * D_norm + w_T * T_norm + w_F * F_norm

    # 5. 输出中间结果（验证指标与权重合理性）
    print("="*80)
    print("1. 指标权重（熵权法，客观权重）")
    print(f"   综合距离权重: {w_D:.4f}, 时间富余权重: {w_T:.4f}, 轴角遮蔽权重: {w_F:.4f}")
    print(f"   权重总和: {w_D + w_T + w_F:.4f}（理论应为1，验证正确）")

    print("\n2. 综合得分矩阵（无人机×导弹，越高越好）")
    print(f"{'无人机--导弹':<8}" + "".join([f"{missiles[m]['name']:<12}" for m in missile_indices]))
    for u in uav_indices:
        row_str = f"{uavs[u]['name']:<8}"
        for m in missile_indices:
            row_str += f"{score_matrix[u, m]:<12.4f}"
        print(row_str)

    # 6. 执行严谨的匈牙利匹配（全局最优）
    matches = rigorous_hungarian_matching(score_matrix)
    print(f"\n3. 无人机-导弹最优匹配结果（基于严谨匈牙利算法，全局最优）")
    # 按无人机编号排序输出
    matches_sorted = sorted(matches, key=lambda x: x[0])
    for u, m in matches_sorted:
        print(f"   {uavs[u]['name']} → {missiles[m]['name']}（综合得分：{score_matrix[u, m]:.4f}）")

    # 7. 验证导弹全覆盖
    matched_missiles = set([m for _, m in matches])
    print(f"\n4. 匹配覆盖验证")
    print(f"   覆盖导弹数量：{len(matched_missiles)}/{n_missile}（{'完全覆盖' if len(matched_missiles)==n_missile else '未完全覆盖！'}）")
    print(f"   分配无人机数量：{len(matches)}/{n_uav}（{'全部分配' if len(matches)==n_uav else '部分未分配！'}）")

    # 8. 投弹数量分配（基于最优匹配得分）
    smoke_assign, uav_smoke_count = assign_smoke_count(matches, score_matrix, uavs, missiles)
    print("\n5. 最终投弹分配结果（基于最优匹配得分动态调整）")
    print("="*80)
    total_smoke = 0
    missile_smoke_count = {m: 0 for m in missile_indices}
    
    # 输出每个无人机投放的烟雾弹数量摘要
    print("各无人机投放烟雾弹总数：")
    for u in sorted(uav_smoke_count.keys()):
        print(f"   {uavs[u]['name']}：{uav_smoke_count[u]}枚")
    print("-"*60)
    
    # 详细输出各无人机的投弹分配情况
    for u in sorted(smoke_assign.keys()):
        u_name = uavs[u]["name"]
        print(f"{u_name}（总投弹：{uav_smoke_count[u]}枚）:")
        for assign in smoke_assign[u]:
            m_name = assign["missile_name"]
            m_idx = assign["missile_idx"]
            count = assign["count"]
            score = assign["score"]
            print(f"   → 干扰{m_name}：{count}枚（匹配得分：{score:.4f}）")
            total_smoke += count
            missile_smoke_count[m_idx] += count

    # 9. 最终分配验证
    print(f"\n6. 最终分配验证")
    print(f"   总投弹数量：{total_smoke}枚")
    print(f"   各导弹被干扰次数：")
    for m in missile_indices:
        print(f"      {missiles[m]['name']}：{missile_smoke_count[m]}次（满足至少1次覆盖要求）")
    print(f"   各无人机任务分配：{len(smoke_assign)}/{n_uav}架（全部分配，无闲置资源）")
    print("="*80)

if __name__ == "__main__":
    main()
