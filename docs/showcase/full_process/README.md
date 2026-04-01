# full_process 封面展示说明

本目录用于 GitHub 首页第一屏展示，聚焦“投弹-爆炸-遮蔽”完整过程的 3D 动态可视化。

## 文件清单

- [full_process_3d.gif](full_process_3d.gif)：3D 全流程动态封面图（推荐首页首图）。
- [full_process_3d.png](full_process_3d.png)：3D 全流程高分辨率静态快照。
- [full_process_timeline.png](full_process_timeline.png)：遮蔽状态时间线图。
- [full_process_summary.json](full_process_summary.json)：封面可视化对应参数与遮蔽区间。

## 复现命令

在项目根目录执行：

```powershell
python "2-code/第二问/aesaga第二问.py" --generations 90 --pop-size 28 --pace 420 --output-dir "docs/showcase/second_question" --hero-output-dir "docs/showcase/full_process" --hero-frames 96 --hero-fps 14
```

## 当前摘要（来自 summary）

- 有效遮蔽时长：4.3071 s
- 机速：71.24 m/s
- 投放时刻：0.0570 s
- 起爆时刻：2.4688 s
- 遮蔽区间：2.7119 s 到 7.0190 s

完整字段请查看 [full_process_summary.json](full_process_summary.json)。
