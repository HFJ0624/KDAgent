# ReAct-adapted Experiment Protocol

## Scope frozen before formal scoring

- Method name: `ReAct-adapted` (ReAct 工具型 Agent，任务适配版).
- Backbone: the `deepseek-v4-pro` entry in `configs/models.yaml` (actual API model identifier: `deepseek-v4-pro`).
- SWaT panel: the 20 records in `data/swat_s2s_raw_window_fixed/llm_prompt_cases.jsonl`.
- WADI panel: the 13 records in `outputs/wadi_external_validation_v1/prepared_data/llm_prompt_cases.jsonl`.
- Repetitions: three runs per episode. A run is retained regardless of accuracy.
- Frozen candidate order and ground truth are never modified. Ground truth is loaded only by the offline scorer.

The expected ReAct-adapted output count is `20 x 3 + 13 x 3 = 99` diagnosis records. These are repeated observations of 33 episodes, not 99 independent episodes.

## Agent and tools

The model starts with the dataset name, episode identifier, frozen candidate IDs/names/types/ranks, and tool schemas. It does not initially receive event evidence or retrieved knowledge. Each logical model call emits one JSON action. The environment then executes that tool and appends its observation before the next decision.

Available tools:

1. `get_episode_evidence`: returns the frozen compact numerical/temporal evidence for the current candidate panel.
2. `search_domain_knowledge`: embeds the model-supplied query and executes a real Top-5 query against the existing dataset-specific Chroma collection.
3. `submit_ranking`: validates and commits an ordered list of 1-5 unique in-candidate variables.

The method does not create KDAgent's two independent branches and does not lock the evidence primary at rank 1. Direct submission is permitted and logged. Invalid JSON/actions consume a logical call and receive a deterministic error observation; no label-dependent correction is used.

## Common generation budget

- Temperature: 0.2.
- Maximum completion tokens per logical call: 8192.
- Thinking budget requested from the provider: 2048.
- Maximum logical generation calls per diagnosis: 4.
- Maximum completion-token budget per diagnosis: 32768.
- Request timeout: 120 seconds.
- Maximum network attempts per logical call: 5; network retries are not counted as new logical generations.
- A diagnosis terminates on a valid `submit_ranking`, or fails after the fourth logical call / total completion budget / API failure.

This envelope accommodates KDAgent's observed maximum of four logical calls (up to three evidence-repair calls plus one retrieval call), Serial's maximum of three, and ReAct-adapted's maximum of four. Actual calls, provider-reported prompt/completion/reasoning/total tokens, network attempts, elapsed time, truncation, parsing errors, tool errors, and stopping reason are retained.

## Fairness and reuse

Existing SWaT KDAgent and Serial records are reused only because their model identifier, temperature, per-call token limit, thinking budget, timeout, candidate panel, and three-run protocol are verifiable from saved configuration/logs. WADI KDAgent has only run 1 saved; runs 2-3 and WADI Serial runs 1-3 must be completed under the same settings before an equal-protocol WADI comparison is reported.

ReAct-adapted has the same frozen evidence and dataset-specific knowledge collection available as the corresponding KDAgent/Serial condition. Output validity for ReAct requires only a nonempty, unique, in-candidate ranking of length at most five with the primary equal to rank 1; KDAgent's evidence-authority contract is not imposed on this baseline.

## Scoring and inference

Metrics are Hit@1/3/5, MRR, NDCG@5, completion rate, and valid-output rate on the full panel and the candidate-covered subset. Multi-label episodes count as a Hit@k when any ground-truth variable intersects the first k predictions; MRR uses the earliest ground-truth rank.

For inference, three runs are first averaged within each episode. The primary endpoint is Hit@5. Four paired comparisons form one new Holm family: KDAgent-ReAct and Serial-ReAct separately on SWaT and WADI. The implementation reports paired mean differences, a 20,000-sample episode bootstrap 95% interval, a two-sided exact paired sign-flip p-value, and Holm-adjusted p-values. If one dataset is incomplete, the four-comparison family is explicitly marked incomplete rather than reduced post hoc.

SWaT overlap groups are computed from the saved episode time intervals. WADI overlap groups are independently computed from its saved row intervals. Sensitivity summaries first average records within each connected overlap component.

## Label isolation and failure policy

`gt_vars`, `ground_truth`, attack answers, and hit fields are absent from the model prompt and every tool observation. The online runner keeps labels only in a separate scoring payload written after the diagnosis completes. Missing, malformed, truncated, timed-out, over-budget, or tool-failed diagnoses remain in the formal panel as misses; they are never silently deleted or replaced with the frozen numerical rank.

## ReAct reference and adaptation

Reference repository: `https://github.com/ysymyth/ReAct`, branch `master`, observed commit `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9` (accessed 2026-09-12). It accompanies Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models,” ICLR 2023. We retain the alternating decision/action, environment execution, observation, and continued-decision loop. The original task environments are replaced by industrial episode-evidence, domain-retrieval, and ranking-submission tools. This is a task adaptation, not an official industrial RCA implementation from that repository.

