"""Generate a minimal synthetic llm_prompt_cases.jsonl for smoke testing."""
import json
import os

cases = []
for i in range(3):
    top10 = [f"V{i*5+j:03d}" for j in range(1, 11)]
    details = [
        {
            "var": v,
            "type": "流量" if j % 2 == 0 else "压力",
            "score": round(0.95 - j * 0.08, 3),
            "raw": round(10 + j, 3),
            "recon": round(10 + j + (0.5 if j == 1 else 0), 3),
            "residual": round(0.5 if j == 1 else 0.05, 3),
            "semantic": f"传感器 {v} 表现出异常行为" + ("（最可能的根因）" if j == 1 else ""),
        }
        for j, v in enumerate(top10, start=1)
    ]
    prompt = (
        f"你是一名 RCA 工程师。案例 {i} 的 Top-10 候选变量如下：\n"
        + "\n".join(
            f"{j}. 变量：{d['var']} | 类型：{d['type']} | 分数：{d['score']} | 原始：{d['raw']} | 重构：{d['recon']} | 残差：{d['residual']} | 语义：{d['semantic']}"
            for j, d in enumerate(details, start=1)
        )
        + "\n\n请返回包含 predicted_root_causes、primary_root_cause、reasoning、evidence_variables、confidence 的 JSON。"
    )
    cases.append(
        {
            "case_id": i,
            "gt_vars": [top10[0]],
            "top10_vars": top10,
            "top10_details": details,
            "prompt": prompt,
        }
    )

out = os.path.join(os.path.dirname(__file__), "..", "data", "llm_prompt_cases.sample.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print("已写入：", out, "案例数：", len(cases))
