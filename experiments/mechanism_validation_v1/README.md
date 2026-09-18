# KDAgent Mechanism Validation v1

本目录补充三项机制验证，不修改原 SWaT 主实验、候选 Top-10、ground truth、知识库或已保存分支输出。

## 运行环境

在 `D:\workspace\lab_TARCA_S2S\llm_rca_experiment` 下运行以下命令。

### 数据核查

```powershell
python -m experiments.mechanism_validation_v1.run_mechanism_validation audit
```

### 实验 1 与实验 2 离线分析

```powershell
python -m experiments.mechanism_validation_v1.run_mechanism_validation offline
```

### 实验 3 dry-run

```powershell
python -m experiments.mechanism_validation_v1.run_mechanism_validation feedback
```

### 一次完成核查、离线分析和 dry-run

```powershell
python -m experiments.mechanism_validation_v1.run_mechanism_validation all
```

上述四条命令均不会调用 LLM。输出默认写入 `outputs\mechanism_validation_v1`。

## 实验 3 正式调用

正式调用必须在用户明确授权后同时打开两个门禁：

```powershell
python -m experiments.mechanism_validation_v1.run_mechanism_validation feedback --execute --allow-llm
```

若运行中断，只能使用断点续跑，避免重复付费：

```powershell
python -m experiments.mechanism_validation_v1.run_mechanism_validation feedback --execute --allow-llm --resume
```

正式结果采用 append-only checkpoint，并分别记录逻辑生成调用和底层网络尝试。已有 checkpoint 时，不带 `--resume` 的正式运行会主动停止。
