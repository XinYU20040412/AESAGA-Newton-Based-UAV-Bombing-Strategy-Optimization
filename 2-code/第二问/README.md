# 第二问 — 单机单弹优化（AESAGA）

本目录求解“FY1 投放 1 枚烟幕弹干扰 M1，使有效遮蔽时长最大化”的连续优化问题。

## 核心脚本

- `aesaga第二问.py`：主优化入口（推荐），支持自动导出收敛图、流程图、GIF、3D 全流程封面。
- `2.py`：早期实验版本，用于对照。

## 优化变量

- 无人机飞行速度。
- 无人机飞行方向角。
- 烟幕弹投放时刻。
- 起爆延迟（或起爆时刻）。

## 运行示例

### 只跑优化

```powershell
python "2-code/第二问/aesaga第二问.py" --skip-gif --skip-hero
```

### 生成 GitHub 展示素材（推荐）

```powershell
python "2-code/第二问/aesaga第二问.py" --generations 90 --pop-size 28 --pace 420 --output-dir "docs/showcase/second_question" --hero-output-dir "docs/showcase/full_process" --hero-frames 96 --hero-fps 14
```

## 主要参数

- `--generations`：迭代代数。
- `--pop-size`：种群规模。
- `--pace`：每次 fitness 评估的时间采样密度。
- `--output-dir`：第二问常规展示产物目录。
- `--skip-gif`：不导出收敛 GIF。
- `--hero-output-dir`：3D 全流程封面产物目录。
- `--skip-hero`：不导出 3D 全流程封面。
- `--hero-frames`：3D 动画帧数。
- `--hero-fps`：3D 动画帧率。

## 输出产物

### 常规展示（`output-dir`）

- `aesaga_optimization.png`：收敛曲线图。
- `aesaga_optimization.gif`：收敛过程动画。
- `aesaga_workflow.png`：算法流程图。
- `aesaga_metrics.csv`：逐代最优/平均适应度。
- `aesaga_summary.json`：最佳参数与摘要。
- `aesaga_best_result.txt`：文本版结果。

### 封面展示（`hero-output-dir`）

- `full_process_3d.gif`：投弹全流程 3D 动态封面。
- `full_process_3d.png`：3D 静态快照。
- `full_process_timeline.png`：遮蔽时间线图。
- `full_process_summary.json`：封面展示对应参数与遮蔽区间。

封面说明见 [docs/showcase/full_process/README.md](../../docs/showcase/full_process/README.md)。
