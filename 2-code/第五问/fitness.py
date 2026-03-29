import numpy as np
from cover_checker import AdvancedMissileSmokeChecker
from system_at_t import cover_system

def init_theta(cover_plan):
    FY = [[17800, 0, 1800], [12000, 1400, 1400], [6000, -3000, 700], [11000, 2000, 1800], [13000, -2000, 1300]]
    M_int = [[20000, 0, 2000], [19000, 600, 2100], [18000, -600, 1900]]
    theta_list = []

    for i in range(len(cover_plan)):
        j = cover_plan[i]
        Xfy = FY[i][0]
        Yfy = FY[i][1]
        XG = (Xfy + M_int[j][0] + 0) / 3
        YG = (Yfy + M_int[j][1] + 200) / 3
        xx = XG - Xfy
        yy = YG - Yfy
        l = (xx ** 2 + yy ** 2) ** 0.5
        cos = xx / l if l > 0 else 1
        theta = np.arccos(cos)
        if yy < 0:
            theta = 2 * np.pi - theta
        theta_list.append(theta)

    return np.array(theta_list)


def pro5_fitness(optim,x:np.ndarray) -> float:
    smoke_sum=0
    vi,theta = [],[]
    tdropik,texplik = np.full((5, 3), 100000),np.full((5, 3), 100000)
    for i in range(5):
        vi.append(float(110 + np.array([x[i] * 60 - 30])))  # 速度范围:70-140
        theta.append(float(optim.theta_base[i] + np.array([(x[i+5] * 30 - 15) * np.pi / 180])))

        # 调整时间参数，确保三个烟雾弹有足够的时间间隔和作用窗口
        delta_ts = [15,15,15]  # 第一个烟雾弹投放时间范围扩大

        for j in range(optim.smoke_num[i]):
            if j==0:
                tdropik[i][j] = x[10+smoke_sum]*delta_ts[0]  # 投放时间:0-15
            else:
                tdropik[i][j] = x[10+smoke_sum] * delta_ts[j] + 1 + tdropik[i][j-1]
            texplik[i][j] = tdropik[i][j] + x[10+smoke_sum+np.sum(optim.smoke_num)] * 10  # 爆炸时间:投放后0-20秒
            smoke_sum+=1

    cover = cover_system(vi, theta, tdropik, texplik)

    # 计算阻断时间 - 提高采样精度
    pace = 300  # 提高采样精度，确保捕捉所有烟雾作用时间
    t_block = 0
    delta_t = 67 / pace  # 延长时间范围至80秒，确保覆盖所有烟雾作用时间

    for t in np.linspace(0, 67, pace):
        Mj, smokes_location = cover(t, 3)
        if len(smokes_location) > 0 and optim.checker.check(Mj, smokes_location):
            t_block += delta_t

    return t_block

