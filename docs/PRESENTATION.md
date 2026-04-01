# GitHub 展示指南（PRESENTATION）

本项目已具备主页级展示能力，建议按下面结构组织你的公开仓库首页。

## 1. 顶部第一屏

- 放置动态封面：docs/showcase/full_process/full_process_3d.gif
- 放置 2-3 行定位文案：问题背景、方法亮点、可复现声明。
- 放置跳转按钮：论文 PDF、题面 PDF、代码总览、展示产物。

建议第一屏口径：

- 这是完整投弹与遮蔽过程的 3D 动态复现实录。
- 动态展示由第二问最优参数驱动，不是手工绘制示意图。

## 2. 第二屏：结果可信度

- 给出问题一到问题五的摘要表。
- 单独注明“展示复现实验”和“论文最终配置”的区别，避免指标歧义。
- 关键数据来源统一指向：
	- 1-paper/附件/result1.xlsx
	- 1-paper/附件/result2.xlsx
	- 1-paper/附件/result3.xlsx
	- docs/showcase/second_question/aesaga_summary.json

## 3. 第三屏：方法可解释性

- 展示流程图：docs/showcase/second_question/aesaga_workflow.png
- 展示静态收敛图：docs/showcase/second_question/aesaga_optimization.png
- 展示全流程时间线：docs/showcase/full_process/full_process_timeline.png
- 给出“如何复现”一条命令。

## 4. 第四屏：结构化导航

- 论文入口：1-paper/README.md
- 代码入口：2-code/README.md
- 文档入口：docs/README.md
- 参考文献入口：3-参考文献/README.md

## 5. 发布前检查

1. 首页所有链接可点击。
2. 首页所有图片均可加载。
3. 所有核心子目录都含 README.md。
4. 复现命令在新环境中可跑通。
