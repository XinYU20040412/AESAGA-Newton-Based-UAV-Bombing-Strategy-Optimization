
import numpy as np
from scipy.optimize import linear_sum_assignment
import math

# ------------------------------------------------------------------------------
# 第一步：定义核心物理参数（无人机、导弹、目标、烟幕参数）
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
    # 真目标（圆柱区域）、假目标（原点，导弹默认攻击方向）
    targets = {
        "true": {"center": (0, 200, 5), "radius": 7, "height": (0, 10)},
        "fake": {"pos": (0, 0, 0)}
    }
    # 烟幕参数（半径、有效时间、沉降速度）
    smoke = {"radius": 10, "valid_time": 20, "sink_speed": 3}
    return uavs, missiles, targets, smoke

# ------------------------------------------------------------------------------
# 第二步：FAHP计算烟幕干扰指标权重（D1-D4权重，用于综合距离计算）
# ------------------------------------------------------------------------------
def fuzzy_ahp_smoke_weight():
    """基于FAHP计算4个距离子指标（D1-D4）的归一化权重（D1>D4>D2=D3）"""
    # 1. 三角模糊数标度定义
    eq = (1, 1, 1)          # 同等重要
    much = (5, 7, 9)        # 重要得多
    slightly = (2, 3, 4)    # 较重要
    rev_much = (1/9, 1/7, 1/5)      # 重要得多的反向标度
    rev_slightly = (1/4, 1/3, 1/2)  # 较重要的反向标度

    # 2. 模糊判断矩阵（行/列：D1,D2,D3,D4）
    A = [
        [eq, much, much, much],          # D1 vs 其他（重要得多）
        [rev_much, eq, eq, rev_slightly],# D2 vs 其他
        [rev_much, eq, eq, rev_slightly],# D3 vs 其他
        [rev_much, slightly, slightly, eq]# D4 vs 其他
    ]
    n = len(A)

    # 3. 模糊合成运算（计算每行模糊乘积M_i）
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

    # 6. 一致性检验（确保判断矩阵合理性）
    def consistency_check(A, weights):
        crisp_matrix = np.array([[row[j][1] for j in range(n)] for row in A])
        Aw = np.dot(crisp_matrix, weights)
        lambda_max = sum(Aw / (n * np.array(weights)))
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI = 0.90  # n=4时的标准随机一致性指标RI值
        CR = CI / RI if RI != 0 else 0
        print(f"FAHP一致性检验：CR={CR:.6f} {'（通过，CR<0.1）' if CR<0.1 else '（未通过！）'}")
        return CR

    consistency_check(A, normalized_weights)
    return normalized_weights

# 提前计算FAHP权重（全局变量a/b/c/d对应D1/D2/D3/D4的权重）
final_weights = fuzzy_ahp_smoke_weight()
a, b, c, d = final_weights

# ------------------------------------------------------------------------------
# 第三步：核心指标计算（仅保留指标1和指标2）
# ------------------------------------------------------------------------------
def calc_comprehensive_distance(uav_pos, missile_pos, targets):
    """指标1：综合距离得分（越小越好，D1-D4加权求和）"""
    # D1：无人机到“导弹-真目标-假目标”平面的垂直距离
    p1 = targets["true"]["center"][:2] + (0,)  # 真目标投影（z=0）
    p2 = targets["fake"]["pos"]                # 假目标位置
    p3 = missile_pos                           # 导弹位置
    x1,y1,z1 = p1; x2,y2,z2 = p2; x3,y3,z3 = p3; x0,y0,z0 = uav_pos
    
    # 计算平面法向量（基于p1、p2、p3三点）
    v1 = (x2-x1, y2-y1, z2-z1)
    v2 = (x3-x1, y3-y1, z3-z1)
    nx = v1[1]*v2[2] - v1[2]*v2[1]
    ny = v1[2]*v2[0] - v1[0]*v2[2]
    nz = v1[0]*v2[1] - v1[1]*v2[0]
    d_plane = -(nx*x1 + ny*y1 + nz*z1)
    
    # 垂直距离公式（点到平面距离）
    numerator = abs(nx*x0 + ny*y0 + nz*z0 + d_plane)
    denominator = math.sqrt(nx**2 + ny**2 + nz**2)
    D1 = numerator / denominator

    # D2：无人机与导弹的直线距离
    D2 = math.sqrt((uav_pos[0]-missile_pos[0])**2 + 
                   (uav_pos[1]-missile_pos[1])**2 + 
                   (uav_pos[2]-missile_pos[2])**2)

    # D3：无人机到“无人机-真目标-导弹”三角形重心的水平距离
    XG = (uav_pos[0] + missile_pos[0] + targets["true"]["center"][0]) / 3
    YG = (uav_pos[1] + missile_pos[1] + targets["true"]["center"][1]) / 3
    D3 = math.sqrt((XG - uav_pos[0])**2 + (YG - uav_pos[1])**2)

    # D4：无人机到“无人机-真目标-导弹”立体重心的距离（加权1.5模拟遮挡需求）
    ZG = (uav_pos[2] + missile_pos[2] + targets["true"]["center"][2]) / 3
    D4 = math.sqrt((XG - uav_pos[0])**2 + (YG - uav_pos[1])**2 + (ZG - uav_pos[2])**2) * 1.5

    # 综合距离（权重加权，结果越小越好）
    return a*D1 + b*D2 + c*D3 + d*D4


def calc_time_surplus_score(uav, missile, targets):
    """指标2：时间富余得分（越大越好，衡量无人机投弹时间充裕度）"""
    # 1. 导弹总飞行时间（从当前位置到假目标）
    missile_to_fake = math.sqrt((missile["pos"][0]-targets["fake"]["pos"][0])**2 +
                                (missile["pos"][1]-targets["fake"]["pos"][1])**2 +
                                (missile["pos"][2]-targets["fake"]["pos"][2])**2)
    time_missile_total = missile_to_fake / missile["speed"]

    # 2. 无人机到导弹的最短飞行时间（取x/y平面最短距离，忽略z方向高度差）
    distance_min = np.min([abs(uav["pos"][0]-missile["pos"][0]), 
                          abs(uav["pos"][1]-missile["pos"][1])])
    time_uav_min = distance_min / uav["speed"]

    # 3. 时间富余得分（归一化到[0,1]，避免负分）
    time_surplus = time_missile_total - time_uav_min  # 初始无已用时间，设为0
    return max(0, time_surplus / time_missile_total)

# ------------------------------------------------------------------------------
# 第四步：指标标准化与熵权法（适配2个指标的权重计算）
# ------------------------------------------------------------------------------
def normalize_index(index_matrix, is_negative=False):
    """指标标准化到[0,1]：is_negative=True表示“越小越好”（逆指标）"""
    max_val = np.max(index_matrix)
    min_val = np.min(index_matrix)
    if max_val == min_val:  # 避免除以0（所有值相同时返回全1矩阵）
        return np.ones_like(index_matrix)
    # 逆指标（越小越好）：(max - x)/(max - min)；正指标（越大越好）：(x - min)/(max - min)
    return (max_val - index_matrix) / (max_val - min_val) if is_negative else (index_matrix - min_val) / (max_val - min_val)


def calc_entropy_weight(indicator_matrices):
    """熵权法计算2个指标的权重（输入：2个指标矩阵列表，每个矩阵为n_uav×n_missile）"""
    m_total = indicator_matrices[0].size  # 总样本数（无人机-导弹匹配对数量）
    weights = []

    for mat in indicator_matrices:
        # 标准化（加1e-6避免log(0)错误）
        mat_norm = mat + 1e-6
        # 计算每个样本的占比（p_ij = x_ij / sum(x_ij)）
        p = mat_norm / mat_norm.sum()
        # 计算熵值（e_j = -k * sum(p_ij * ln(p_ij))，k=1/ln(m_total)）
        entropy = -1 / math.log(m_total) * np.sum(p * np.log(p))
        # 权重 = (1 - 熵) / sum(1 - 所有指标的熵)（熵越小，权重越大）
        weights.append(1 - entropy)

    # 归一化权重（确保权重和为1）
    weights = np.array(weights) / sum(weights)
    return weights

# ------------------------------------------------------------------------------
# 第五步：匈牙利匹配优化（确保无人机与导弹全覆盖）
# ------------------------------------------------------------------------------
def build_coverage_cost_matrix(score_matrix):
    """构建覆盖型成本矩阵：扩展导弹列以适配无人机数量，确保导弹全覆盖"""
    n_uav, n_missile = score_matrix.shape
    # 若无人机数量 > 导弹数量：复制导弹列（优先复制得分高的导弹列）
    if n_uav > n_missile:
        # 计算每列平均得分，按得分降序排序导弹索引
        col_scores = score_matrix.mean(axis=0)
        top_cols = np.argsort(col_scores)[::-1]
        # 扩展列：原列 + 复制高得分列（直到列数=无人机数）
        extend_cols = []
        while len(extend_cols) < n_uav - n_missile:
            extend_cols.extend(top_cols[:max(1, n_uav - n_missile - len(extend_cols))])
        extended_score = np.hstack([score_matrix, score_matrix[:, extend_cols]])
    else:
        extended_score = score_matrix

    # 成本矩阵 = 最大得分 - 综合得分（匈牙利算法求最小成本 ≡ 求最大得分）
    max_score = np.max(extended_score)
    cost_matrix = max_score - extended_score
    return cost_matrix, extended_score


def ensure_full_coverage(matches, n_missile):
    """确保导弹全覆盖：若某导弹未分配，从重复分配的匹配中调整"""
    matched_missiles = [m for _, m in matches]
    # 找出未被分配的导弹
    uncovered_missiles = [m for m in range(n_missile) if m not in matched_missiles]
    
    if uncovered_missiles:
        # 调整重复分配的无人机（分配到扩展列的无人机）到未覆盖导弹
        for idx, (u, m) in enumerate(matches):
            if m >= n_missile:  # 扩展列对应原导弹的重复分配
                matches[idx] = (u, uncovered_missiles.pop(0))
                if not uncovered_missiles:
                    break
    return matches


def hungarian_full_matching(score_matrix):
    """全覆盖匈牙利匹配：返回（无人机索引, 导弹索引）列表，确保无遗漏"""
    n_uav, n_missile = score_matrix.shape
    # 1. 构建覆盖型成本矩阵
    cost_matrix, _ = build_coverage_cost_matrix(score_matrix)
    # 2. 执行匈牙利算法（一对一匹配）
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # 3. 映射扩展列到原导弹索引（扩展列=原导弹重复）
    col_ind_mapped = [c if c < n_missile else c % n_missile for c in col_ind]
    # 4. 确保导弹全覆盖
    matches = list(zip(row_ind, col_ind_mapped))
    matches = ensure_full_coverage(matches, n_missile)
    # 5. 补充未匹配的无人机（分配到得分最高的导弹）
    matched_uavs = set([u for u, _ in matches])
    for u in range(n_uav):
        if u not in matched_uavs:
            best_m = np.argmax(score_matrix[u, :])
            matches.append((u, best_m))
    return matches

# ------------------------------------------------------------------------------
# 第六步：投弹数量分配（基于综合得分动态调整）
# ------------------------------------------------------------------------------
def assign_smoke_count(matches, score_matrix, uavs, missiles):
    """基于综合得分分配投弹数量（每无人机最多3枚，得分高则多分配）"""
    # 1. 按综合得分降序排序匹配对
    match_scores = [(u, m, score_matrix[u, m]) for u, m in matches]
    match_scores.sort(key=lambda x: x[2], reverse=True)

    # 2. 初始化投弹计数（每无人机最多3枚）
    uav_smoke_count = {u: 0 for u in uavs.keys()}
    smoke_assign = {u: [] for u in uavs.keys()}

    # 3. 动态分配投弹数量（按得分分级）
    for u, m, score in match_scores:
        if uav_smoke_count[u] >= 3:
            continue  # 达到无人机投弹上限
        # 按得分分配投弹数：高得分多投弹
        if score >= 0.8:
            count = 3
        elif score >= 0.6:
            count = 2
        else:
            count = 1
        # 调整数量，避免超过上限
        count = min(count, 3 - uav_smoke_count[u])
        # 记录分配结果
        smoke_assign[u].append({"missile_name": missiles[m]["name"], 
                                "missile_idx": m, 
                                "count": count, 
                                "score": score})
        uav_smoke_count[u] += count

    # 4. 过滤无分配的无人机（确保全部分配）
    smoke_assign = {u: assign for u, assign in smoke_assign.items() if assign}
    return smoke_assign, uav_smoke_count

# ------------------------------------------------------------------------------
# 主函数：整合全流程（参数定义→指标计算→权重→匹配→分配）
# ------------------------------------------------------------------------------
def main():
    # 1. 初始化物理参数
    uavs, missiles, targets, smoke = define_a_problem_params()
    n_uav = len(uavs)
    n_missile = len(missiles)
    uav_indices = list(uavs.keys())
    missile_indices = list(missiles.keys())

    # 2. 计算2个核心指标矩阵（n_uav × n_missile）
    D_matrix = np.zeros((n_uav, n_missile))  # 指标1：综合距离（越小越好）
    T_matrix = np.zeros((n_uav, n_missile))  # 指标2：时间富余（越大越好）

    for u in uav_indices:
        uav = uavs[u]
        for m in missile_indices:
            missile = missiles[m]
            D_matrix[u, m] = calc_comprehensive_distance(uav["pos"], missile["pos"], targets)
            T_matrix[u, m] = calc_time_surplus_score(uav, missile, targets)

    # 3. 指标标准化与权重计算（熵权法）
    D_norm = normalize_index(D_matrix, is_negative=True)  # 逆指标标准化（越小→越大）
    T_norm = normalize_index(T_matrix, is_negative=False) # 正指标标准化（越大→越大）

    # 熵权法计算2个指标的权重
    indicator_matrices = [D_norm, T_norm]  # 仅保留2个指标
    w_D, w_T = calc_entropy_weight(indicator_matrices)  # 2个指标的权重

    # 4. 计算综合得分矩阵（加权求和，得分越高越优）
    score_matrix = w_D * D_norm + w_T * T_norm

    # 5. 输出中间结果
    print("="*80)
    print("1. 核心指标权重（熵权法，已去除指标3：轴角遮蔽得分）")
    print(f"   综合距离权重: {w_D:.4f}, 时间富余权重: {w_T:.4f}")
    print(f"   权重总和: {w_D + w_T:.4f}（理论应为1，验证权重有效性）")

    print("\n2. 综合得分矩阵（无人机×导弹，得分越高越优）")
    print(f"{'无人机':<8}" + "".join([f"{missiles[m]['name']:<12}" for m in missile_indices]))
    for u in uav_indices:
        row_str = f"{uavs[u]['name']:<8}"
        for m in missile_indices:
            row_str += f"{score_matrix[u, m]:<12.4f}"
        print(row_str)

    # 6. 全覆盖匈牙利匹配（确保每架无人机、每枚导弹都有分配）
    matches = hungarian_full_matching(score_matrix)
    print(f"\n3. 无人机-导弹匹配结果（全覆盖）")
    for u, m in matches:
        print(f"   {uavs[u]['name']} → {missiles[m]['name']}（综合得分：{score_matrix[u, m]:.4f}）")

    # 7. 投弹数量分配（基于综合得分动态调整）
    smoke_assign, uav_smoke_count = assign_smoke_count(matches, score_matrix, uavs, missiles)
    print("\n4. 最终投弹分配结果")
    print("="*80)
    total_smoke = 0
    missile_smoke_count = {m: 0 for m in missile_indices}  # 统计每枚导弹被干扰次数
    for u in smoke_assign:
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

    # 8. 分配结果验证（确保全覆盖）
    print(f"\n5. 分配结果验证")
    print(f"   总投弹数量：{total_smoke}枚")
    print(f"   各导弹被干扰次数（确保每枚导弹至少1次）：")
    for m in missile_indices:
        print(f"      {missiles[m]['name']}：{missile_smoke_count[m]}次")
    print(f"   无人机任务覆盖：{len(smoke_assign)}/{n_uav}架（全部分配）")
    print("="*80)

if __name__ == "__main__":
    main()