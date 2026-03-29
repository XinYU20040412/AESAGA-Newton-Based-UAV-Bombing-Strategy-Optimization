# 复现指南（REPRODUCTION）

本说明列出按顺序复现论文主要结果的建议步骤。建议在安装好 `environment.yml` 指定的环境后开始。

1. 环境准备
```powershell
conda env create -f environment.yml
conda activate hnu_aesaga
pip install -r requirements.txt
```

2. 问题 1（快速）
```powershell
python "第一问/1.py"
```

3. 问题 2（优化）
- 运行 `第二问/aesaga第二问.py`。默认参数可能运行较久，建议先编辑脚本减少 `population` 与 `generations` 测试正确性。

4. 问题 3、4、5
- 参照各子目录 README 中的运行说明，先在小规模参数下验证脚本能运行，再放开参数做完整实验。

5. 注意事项
- 若某脚本依赖 Excel 输入/输出，请确保脚本中设置的路径存在并具有写权限。  
- 若需要并行化或长时间运行，建议在具有良好 CPU 资源的环境中执行或在脚本中加入日志记录以便中断后续处理。
