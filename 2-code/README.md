# 2-code 代码总览

本目录按题号组织了完整求解代码，覆盖问题一到问题五及方法对比实验。

## 子目录导航

- [第一问](第一问/README.md)：单无人机单弹基础模型。
- [第二问](第二问/README.md)：单机单弹时空优化（AESAGA 重点展示模块）。
- [第三问](第三问/README.md)：单机多弹协同优化。
- [第四问](第四问/README.md)：多机协同投放优化。
- [第五问](第五问/README.md)：两阶段策略（匹配 + 局部优化）。
- [第五问 - 不同方法实验](第五问%20-不同方法实验/README.md)：方法对比与消融实验。

## 模块分层逻辑

- 问题一：给定参数下的基础物理仿真与遮蔽计算。
- 问题二：在问题一模型上做参数优化（AESAGA），并自动导出 GitHub 展示素材。
- 问题三：扩展到单机多弹协同时序优化。
- 问题四：扩展到多机协同策略优化。
- 问题五：扩展到多机多弹多导弹综合决策（匹配 + 局部优化）。
- 不同方法实验：用于对比不同算法和参数策略。

## 推荐阅读路径

1. 从第二问开始，先看优化主干逻辑和可视化导出。
2. 再看第三问和第四问，理解单机到多机的扩展。
3. 最后看第五问，理解多目标匹配与全局策略。

## 运行建议

- 复现完整结果前，建议先运行小规模参数验证脚本正确性。
- 正式展示推荐使用第二问导出的可视化产物，路径见 [docs/showcase/second_question/README.md](../docs/showcase/second_question/README.md)。
- 首页封面级展示推荐使用 [docs/showcase/full_process/README.md](../docs/showcase/full_process/README.md)。

## 最小展示命令

```powershell
python "2-code/第二问/aesaga第二问.py" --generations 90 --pop-size 28 --pace 420 --output-dir "docs/showcase/second_question" --hero-output-dir "docs/showcase/full_process"
```
