<<<<<<< HEAD
# 基于 AESAGAWithNewton 的无人机投弹策略优化（2025国赛A）

> 说明：此仓库保留作者提交的原始代码（未修改）。本次补充主要为 GitHub 展示准备文档，包含快速复现与每个子目录的用途说明。

## 项目简介
本项目围绕“无人机投放烟幕干扰弹，最大化对固定圆柱形真目标的有效遮蔽时间”问题展开。采用 AESAGAWithNewton 混合优化算法（改进的遗传算法 + 牛顿局部搜索）求解一系列子问题，从单机单弹到多机多弹、匹配分配与局部优化，最终给出可复现的实验结果（见 result1.xlsx / result2.xlsx / result3.xlsx）。

## 关键成果（论文摘要）
- 问题1：单无人机、单弹基础模型，计算遮蔽时长 ~ **1.39 s**。
- 问题2：单机单弹时空优化，最优遮蔽时长 **4.69 s**（见论文与脚本输出）。
- 问题3：单机三弹协同，最优遮蔽时长 **5.81 s**（`result1.xlsx`）。
- 问题4：三机协同投放，最优遮蔽时长 **8.04 s**（`result2.xlsx`）。
- 问题5：多机多弹对多导弹的两阶段优化（匹配 + 局部优化），结果见 `result3.xlsx`。

## 快速开始（推荐）
建议使用 Miniconda 创建干净环境（示例环境文件 `environment.yml` 已添加）：

```powershell
conda env create -f environment.yml
conda activate hnu_aesaga
pip install -r requirements.txt
```

如果你已经有 `sanjin` 环境（或其它 Python 3.10 环境），也可以直接安装依赖：
```powershell
conda activate sanjin
pip install -r requirements.txt
```

> 提示：完整优化可能耗时，建议在初次运行时先在脚本中调低种群数与迭代代数以快速验证。

## 仓库结构（概览）
- `第一问/` — 单无人机单弹模型，入口：`第一问/1.py`
- `第二问/` — 单机单弹的时空优化，入口：`第二问/aesaga第二问.py` 或 `第二问/2.py`
- `第三问/` — 单机多弹（多个版本的实现），入口见子目录说明
- `第四问/` — 多机协同（入口：`第四问/4源代码.py` 等）
- `第五问/` — 两阶段匹配 + 局部优化，输出 `result3.xlsx`
- `第五问 -不同方法实验/` — 对比实验脚本
- `参考文献/` — 支撑参考资料与原始文献
- `docs/` — 复现指南与展示建议

每个子目录均包含单独的 `README.md`（已补充），请从对应子目录查看更详细的运行说明。

## 运行示例（快速验证）
1. 问题一快速运行（较快）：
```powershell
python "第一问/1.py"
```
2. 问题二示例（优化，耗时）：
```powershell
python "第二问/aesaga第二问.py"
```

## 结果文件
- `result1.xlsx`, `result2.xlsx`, `result3.xlsx`：问题 3、4、5 的输出结果（若存在）。

## 如何在 GitHub 上展示（建议）
- 将关键图（收敛曲线、时序图、遮蔽示意）存入 `docs/images/` 并在 README 中引用。  
- 可创建一个 Jupyter Notebook（`notebooks/quick_demo.ipynb`）用来演示 Q1 的可视化步骤，便于浏览者交互查看。  
- 在 README 顶部放置简短的“故事线”与结果摘要，吸引浏览者快速了解贡献。

---

若需我把 README 再美化（添加截图、表格、徽章）或生成 Notebook 演示，我可以继续协助。
=======
# 2025全国大学生数学建模竞赛
本人承担独立建模手与编程手

>>>>>>> ec39e9d217c9f8fbfeebee21f0d28d34093fbe97
