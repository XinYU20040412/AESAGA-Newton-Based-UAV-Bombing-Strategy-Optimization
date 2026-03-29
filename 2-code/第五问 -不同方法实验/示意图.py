import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --------------------------
# 1. 从文档提取坐标数据（源自）
# --------------------------
# 假目标（原点，xy平面为水平面）
fake_target = np.array([[0, 0, 0]])  # 坐标：(0, 0, 0)
# 真目标（下底面圆心）
real_target = np.array([[0, 200, 0]])  # 坐标：(0, 200, 0)（）
# 3枚来袭导弹（M1、M2、M3）
missiles = np.array([
    [20000, 0, 2000],    # M1：(20000, 0, 2000)（）
    [19000, 600, 2100],  # M2：(19000, 600, 2100)（）
    [18000, -600, 1900]  # M3：(18000, -600, 1900)（）
])
# 5架无人机（FY1~FY5）
uavs = np.array([
    [17800, 0, 1800],    # FY1：(17800, 0, 1800)（）
    [12000, 1400, 1400], # FY2：(12000, 1400, 1400)（）
    [6000, -3000, 700],  # FY3：(6000, -3000, 700)（）
    [11000, 2000, 1800], # FY4：(11000, 2000, 1800)（）
    [13000, -2000, 1300] # FY5：(13000, -2000, 1300)（）
])

# --------------------------
# 2. 创建3D图像
# --------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# --------------------------
# 3. 绘制各对象（不同颜色+标记区分）
# --------------------------
# 假目标（红色五角星，原点标记）
ax.scatter(fake_target[:, 0], fake_target[:, 1], fake_target[:, 2], 
           color='red', marker='*', s=200, label='假目标（原点）')
# 真目标（蓝色圆形）
ax.scatter(real_target[:, 0], real_target[:, 1], real_target[:, 2], 
           color='blue', marker='o', s=150, label='真目标')
# 导弹（橙色三角形，标注M1~M3）
ax.scatter(missiles[:, 0], missiles[:, 1], missiles[:, 2], 
           color='orange', marker='^', s=120, label='来袭导弹')
for i, (x, y, z) in enumerate(missiles):
    ax.text(x, y, z, f'M{i+1}', fontsize=10, color='darkred')  # 导弹标签
# 无人机（绿色正方形，标注FY1~FY5）
ax.scatter(uavs[:, 0], uavs[:, 1], uavs[:, 2], 
           color='green', marker='s', s=120, label='无人机')
for i, (x, y, z) in enumerate(uavs):
    ax.text(x, y, z, f'FY{i+1}', fontsize=10, color='darkgreen')  # 无人机标签

# --------------------------
# 4. 设置图像样式（坐标轴、图例、视角等）
# --------------------------
# 坐标轴标签（对应文档坐标系：xy为水平面，z为高度）
ax.set_xlabel('X轴 (m)', fontsize=12)
ax.set_ylabel('Y轴 (m)', fontsize=12)
ax.set_zlabel('Z轴 (高度, m)', fontsize=12)
# 坐标轴范围（适配所有点的坐标范围，避免点重叠或超出视野）
ax.set_xlim([0, 21000])  # X范围：0（假目标）~20000（M1）
ax.set_ylim([-3500, 2500])  # Y范围：-3000（FY3）~1400（FY2）
ax.set_zlim([0, 2200])  # Z范围：0（假/真目标）~2100（M2）
# 图例（放在图像右上角，不遮挡点）
ax.legend(loc='upper right', fontsize=10)
# 初始视角（调整x、y轴旋转角度，便于观察各点位置关系）
ax.view_init(elev=20, azim=45)
# 标题（标注数据来源）
plt.title('烟幕干扰弹投放策略坐标系3D图（数据源自A题.pdf）', fontsize=14, pad=20)

# --------------------------
# 5. 显示或保存图像
# --------------------------
plt.tight_layout()
# 两种选择：1. 显示图像；2. 保存为PNG文件（路径可自行修改）
plt.show()
#plt.savefig('smoke_screen_coordinate_3d.png', dpi=300, bbox_inches='tight')