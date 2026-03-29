import numpy as np
from typing import *

class cover_system:
    def __init__(self,v_i,theta_i,tdrop_ik,texpl_ik):
        self.fy_init=np.array([[17800,0,1800],[12000,1400,1400],[6000,-3000,700],[11000,2000,1800],[13000,-2000,1300]])
        self.v_i = (
        lambda a1, a2: np.pad(a1, (0, len(a2) - len(a1)), mode='constant')
        if len(a1) < len(a2)
        else a1)(v_i, self.fy_init)
        self.theta_i = (
        lambda a1, a2: np.pad(a1, (0, len(a2) - len(a1)), mode='constant')
        if len(a1) < len(a2)
        else a1)(theta_i, self.fy_init)
        self.ui=np.array(list(map(lambda item: self.v_i[item[0]] * np.array([np.cos(item[1]), np.sin(item[1]), 0]),
        enumerate(self.theta_i))
        ))
        self.tdrop_ik = pad_array_to_match(tdrop_ik,self.fy_init,100000)
        self.texpl_ik = pad_array_to_match(texpl_ik,self.fy_init,100000)
        self.M_init = np.array([[20000,0,2000],[19000,600,2100],[18000,-600,1900]])
        self.v_M=np.array([[-298.51,-0,-29.85],[-291.27,-9.2,-32.19],[-282.26,9.41,-29.79]])

    def __call__(self,t,n_missiles):
        Mj = self.M_init + self.v_M*t
        Mj = Mj[Mj[:, 2] > 0]
        smokes_location = []
        class smoke:  # 创建每一颗烟雾弹对象，使其包含一颗烟雾弹的所有信息
            def __init__(self, fy_init, u, tdrop, texpl):
                self.fy_init = fy_init
                self.u = u
                self.tdrop = tdrop
                self.texpl = texpl
                self.P_ik=np.array(fy_init+u*tdrop)
                self.Z_ik=self.P_ik[2]-0.5*9.8*(texpl-tdrop)**2
                self.Q_ik=np.array(self.P_ik+u*(texpl-tdrop))
                self.Q_ik[2]=self.Z_ik

            def __call__(self, time):
                self.Q_ik[2]=self.Q_ik[2]-(time-self.texpl)*3
                return self.Q_ik
        # 遍历烟雾弹爆炸时刻，如果还未爆炸则不输出坐标
        for i, row in enumerate(self.texpl_ik):
            for j, value in enumerate(row):
                # 设置条件,如果t大于起爆时刻则计算烟幕位置
                if t > value and t < value+20:  # 这里可以根据需要修改条件
                    # 创建对象并添加到列表
                    smoke_eff = smoke(self.fy_init[i], self.ui[i], self.tdrop_ik[i][j], value)
                    smoke_location=smoke_eff(t)
                    if smoke_location[2]>0:
                        smokes_location.append(smoke_location)
        return Mj[:n_missiles], np.array(smokes_location)


def pad_array_to_match(
        array_to_pad: np.ndarray,
        target_array: np.ndarray,
        pad_value: Union[int, float]
) -> np.ndarray:#将未释放的烟雾弹的投掷和爆炸时间设为100000使其失效而不进行计算
    """
    使用函数式编程风格将数组填充到目标数组的尺寸

    参数:
    array_to_pad: 需要填充的数组
    target_array: 目标尺寸的数组
    pad_value: 填充值，默认为1000

    返回:
    填充后的数组，如果不需要填充则返回原数组
    """
    # 获取两个数组的形状
    pad_shape = array_to_pad.shape
    target_shape = target_array.shape

    # 检查是否需要填充
    needs_padding = any(pad_dim < target_dim for pad_dim, target_dim in zip(pad_shape, target_shape))

    # 如果不需要填充，直接返回原数组
    if not needs_padding:
        return array_to_pad

    # 计算需要在右下方向填充的数量
    pad_width = tuple(
        (0, max(0, target_dim - pad_dim))
        for pad_dim, target_dim in zip(pad_shape, target_shape)
    )

    # 使用np.pad进行填充
    return np.pad(
        array_to_pad,
        pad_width=pad_width,
        mode='constant',
        constant_values=pad_value
    )

