# 第二问展示产物说明

本目录由 [2-code/第二问/aesaga第二问.py](../../../2-code/第二问/aesaga第二问.py) 自动生成，用于 GitHub 首页展示。

若需要 3D 全流程封面，请同时查看 [../full_process/README.md](../full_process/README.md)。

## 文件清单

- [aesaga_optimization.gif](aesaga_optimization.gif)：优化过程动态收敛图。
- [aesaga_optimization.png](aesaga_optimization.png)：优化收敛静态图。
- [aesaga_workflow.png](aesaga_workflow.png)：AESAGA 求解流程图。
- [aesaga_metrics.csv](aesaga_metrics.csv)：逐代最优/平均适应度。
- [aesaga_summary.json](aesaga_summary.json)：最佳参数与统计摘要。
- [aesaga_best_result.txt](aesaga_best_result.txt)：可读性结果文本。

## 复现命令

在项目根目录执行：

```powershell
python "2-code/第二问/aesaga第二问.py" --generations 90 --pop-size 28 --pace 420 --output-dir "docs/showcase/second_question" --hero-output-dir "docs/showcase/full_process"
```

## 当前关键结果

- best_fitness: 4.307143 s
- speed: 71.24 m/s
- theta: 176.05°
- tdrop: 0.06 s
- texpl: 2.47 s

完整字段见 [aesaga_summary.json](aesaga_summary.json)。
