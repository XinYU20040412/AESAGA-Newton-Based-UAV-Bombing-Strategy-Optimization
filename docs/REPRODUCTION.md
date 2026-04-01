# 复现指南（REPRODUCTION）

本说明用于快速复现论文核心结果，并生成 GitHub 首页可直接展示的动态图与图表。

## 1. 环境准备

```powershell
conda env create -f environment.yml
conda activate hnu_aesaga
pip install -r requirements.txt
```

## 2. 快速验证（问题一）

```powershell
python "2-code/第一问/1.py"
```

## 3. 生成第二问展示产物（推荐）

```powershell
python "2-code/第二问/aesaga第二问.py" --generations 90 --pop-size 28 --pace 420 --output-dir "docs/showcase/second_question" --hero-output-dir "docs/showcase/full_process"
```

执行后将自动生成：

- 动态 GIF：docs/showcase/second_question/aesaga_optimization.gif
- 收敛图：docs/showcase/second_question/aesaga_optimization.png
- 流程图：docs/showcase/second_question/aesaga_workflow.png
- 摘要：docs/showcase/second_question/aesaga_summary.json
- 3D 封面 GIF：docs/showcase/full_process/full_process_3d.gif
- 3D 封面静态图：docs/showcase/full_process/full_process_3d.png
- 全流程时间线：docs/showcase/full_process/full_process_timeline.png
- 全流程摘要：docs/showcase/full_process/full_process_summary.json

## 4. 复现问题三、四、五

- 按各目录 README 顺序执行：
	- 2-code/第三问/README.md
	- 2-code/第四问/README.md
	- 2-code/第五问/README.md

建议先用小规模参数验证，再执行完整配置。

## 5. 结果核对

- 论文结果表：1-paper/附件/result1.xlsx、result2.xlsx、result3.xlsx
- 第二问展示摘要：docs/showcase/second_question/aesaga_summary.json

## 6. 注意事项

- 若脚本依赖 Excel 输入/输出，请确保路径存在且有写权限。
- 长时间优化建议保留日志输出，避免中断后无法追踪进度。
