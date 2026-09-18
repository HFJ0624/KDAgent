# ReAct-adapted v1

该目录实现 ReAct 工具型 Agent（任务适配版）。它保留 ReAct 的“行动决策—工具执行—观察—继续决策”循环，但将工具适配为冻结事件证据、现有领域知识库和候选排名提交。它不是 RCAgent，也不是官方工业 RCA 系统复现。

## 一键命令

在 `D:\workspace\lab_TARCA_S2S\llm_rca_experiment` 下执行：

```powershell
python -m experiments.react_adapted_v1.run_experiment --mode all-offline
```

完成真实 ReAct 99 条记录、WADI 缺失对照、评分与打包：

```powershell
$env:DASHSCOPE_API_KEY="你的密钥"
python -m experiments.react_adapted_v1.run_experiment --mode formal --approve-api --resume
```

分步运行：

```powershell
python -m experiments.react_adapted_v1.run_experiment --mode audit
python -m experiments.react_adapted_v1.run_experiment --mode synth-test
python -m experiments.react_adapted_v1.run_experiment --mode dry-run
python -m experiments.react_adapted_v1.run_experiment --mode run-react --approve-api --resume
python -m experiments.react_adapted_v1.run_experiment --mode run-wadi-comparators --approve-api
python -m experiments.react_adapted_v1.run_experiment --mode score
python -m experiments.react_adapted_v1.run_experiment --mode package
```

正式记录一旦存在，不带 `--resume` 的运行会拒绝覆盖。API Key 只从环境变量读取，不写入配置、日志或压缩包。

