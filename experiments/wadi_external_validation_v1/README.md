# WADI Minimum-Cost External Validation

本目录只实现固定协议下的 WADI 外部验证：

- `DeepSeek-V4-Pro`
- `Baseline` vs. `KDAgent authority-preserving`
- 14 个 Attack Label 连续区间，其中 13 个具有可解析 `root_tags`
- Phase A 为 1 run；仅在 KDAgent 的 Hit@5 或 covered Top-5 recovery 更高时追加 runs 2-3

## 固定数据口径

1. Episode 由攻击文件中的二值 Attack Label 连续区间产生，不使用损坏的 `Time` 字段做行匹配。
2. 14 个连续区间按 `wadi_attack_mapping.csv` 固定行序一一对应，不拆分、不重叠增广。
3. Ground Truth 只采用 `root_tags`；`affected_tags` 只保留在审计清单中。
4. Attack 14 缺少 `root_tags`，保留在 manifest，但不参与指标计算。
5. W000-W126 按正常文件中 127 个公共过程变量的固定列顺序产生。
6. TA-RCA 使用当前 AERCA 项目的 WADI 参数、median-IQR 预处理、5 倍下采样和 first-valid 上下文策略；训练一次后冻结。
7. 5000 训练预算解释为与原单序列 AERCA 对齐的 5000 次 optimizer updates；训练块按固定顺序循环，避免 WADI 分块将更新次数错误放大约 62 倍。80/20 划分与 20 次全量验证早停保持不变。

## 环境

当前系统 Python 必须能够导入 PyTorch、SciPy、scikit-learn 和 tqdm：

```powershell
python -m pip install -r experiments\wadi_external_validation_v1\requirements-wadi.txt
```

模型和 Embedding 均读取现有 `DASHSCOPE_API_KEY`。WADI 使用独立的
`wadi_process_kb_v1` collection，不会覆盖 SWaT 向量库。

## 一键运行

在 `llm_rca_experiment` 目录执行：

```powershell
python -X utf8 -m experiments.wadi_external_validation_v1.run_wadi_external_validation --mode auto --approve-api
```

该命令依次完成：TA-RCA 训练/复用、候选证据导出、WADI KB 构建、固定随机
3-episode sanity check、Phase A、完整性检查、条件式 Phase B 和最终汇总。
中断后执行同一命令即可续跑；已完成的 LLM 记录和冻结 checkpoint 会被复用。

只准备 TA-RCA、暂不调用任何 API：

```powershell
python -X utf8 -m experiments.wadi_external_validation_v1.run_wadi_external_validation --mode prepare
```

只构建 KB 并进行 3-episode sanity check，不运行 LLM：

```powershell
python -X utf8 -m experiments.wadi_external_validation_v1.run_wadi_external_validation --mode sanity --approve-api
```

只运行 Phase A，不自动追加 runs 2-3：

```powershell
python -X utf8 -m experiments.wadi_external_validation_v1.run_wadi_external_validation --mode phase-a --approve-api
```

## 输出

所有新产物位于 `outputs/wadi_external_validation_v1/`，主要文件包括：

- `wadi_episode_manifest.csv`
- `wadi_tarca_top10.json`
- `sanity_check_3_episodes.txt`
- `wadi_baseline_predictions.json`
- `wadi_kdagent_predictions.json`
- `phase_a_metrics.csv`
- `wadi_metrics.csv`
- `table_cross_system_validation_wadi.csv`
- `wadi_result_summary.json`

Baseline 与 KDAgent 的原始响应、token、推理内容和运行日志分别保存在各自子目录中。
