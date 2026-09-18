# KDAgent: LLM Root Cause Analysis Experiment Framework

A local Python framework for comparing large language models (LLMs) on **industrial process Root Cause Analysis (RCA)** using the SWaT testbed as the primary scenario. It evaluates LLM-driven candidate re-ranking under four ablation settings — plain prompting, RAG retrieval-augmented prompting, an iterative self-refinement agent, and their combination — plus an optional dual-branch deterministic-fusion mode (the KDAgent method).

## Highlights

- Reads a unified case format (`llm_prompt_cases.jsonl`) built on top of a TA-RCA Top-10 candidate list and per-window time-series evidence.
- Uses **OpenAI-compatible APIs** so the same pipeline works with DashScope, DeepSeek, Zhipu, GLM, and other vendors (configured in `configs/models.yaml`).
- Configurable prompt template that automatically assembles Top-10 candidate details and optional RAG context.
- Reproducible experiments: repeated inference per case (`--num_runs`), fixed `temperature`, `max_tokens`, and optional reasoning (`thinking_budget`).
- Robust end-to-end pipeline: retries, timeouts, and failure-tolerant calls; robust JSON parsing with regex fallback; deterministic validation of every response against the candidate set.
- Automatic metrics: `Accuracy@1/@3/@5`, `valid_json_rate`, `hallucination_rate`, `average_confidence`, plus Iterative Agent and RAG diagnostic metrics.
- Full audit trail: raw responses (JSONL), flattened parsed results (CSV), and aggregated metrics (JSON/CSV) for every model and condition.
- One-command matrix runner for the official 5 models × 5 conditions ablation study.
- Offline analysis scripts that recompute metrics without calling any LLM API.

## Repository Layout

```text
llm_kdagent_experiment/
├── configs/
│   ├── models.yaml              # Model config (endpoints, API-key env vars, sampling)
│   └── rag.yaml                 # RAG config (embedding, vector store, retrieval)
├── data/
│   ├── llm_prompt_cases.sample.jsonl  # Minimal sample of the case schema
│   └── rag_kb/                  # RAG knowledge-base source markdown files
│       ├── kb_process.md
│       ├── kb_relations.md
│       ├── kb_fault_patterns.md
│       ├── kb_industrial_rules.md
│       └── kb_official_swat_*.md
├── prompts/
│   └── rca_prompt_template.txt  # Prompt template ({process_knowledge}, {top10_candidates})
├── src/
│   ├── main.py                  # Single-model entry point
│   ├── run_all_models.py        # Batch runner across all configured models
│   ├── model_client.py          # LLM API client (retries / backoff / timeout)
│   ├── data_loader.py           # Case loading and normalization
│   ├── prompt_builder.py        # Prompt construction (with optional RAG block)
│   ├── response_parser.py       # Robust JSON parsing with regex fallback
│   ├── response_validator.py    # Structural + content validation rules
│   ├── evaluator.py             # Hit@K / hallucination / metrics computation
│   ├── iterative_agent.py       # Iterative self-refinement agent
│   ├── dual_branch_fusion_agent.py  # Evidence + RAG dual-branch deterministic fusion
│   ├── rag_*.py, build_rag_index.py # RAG embedding / vector store / retrieval
│   ├── utils.py                 # Shared utilities
│   └── test_*.py                # Unit tests
├── scripts/
│   ├── run_final_experiments.py # Official ablation-matrix runner
│   ├── run_retrieval_robustness.py  # Retrieval-robustness experiment
│   ├── make_sample_data.py      # Generate synthetic sample cases
│   ├── smoke_test*.py           # End-to-end smoke tests (mocked LLM)
│   └── verify_csv.py            # Validate parsed CSVs after a run
├── experiments/                 # Additional experiment variants (see below)
├── analysis/                    # Offline metrics recomputation / audit scripts
├── requirements.txt
└── README.md
```

> `outputs/`, `logs/`, large external datasets under `data/`, and the auto-built ChromaDB vector store are excluded from version control via `.gitignore`.

## Installation

Requires Python 3.9+.

```bash
cd llm_kdagent_experiment
pip install -r requirements.txt
```

Key dependencies: `openai`, `chromadb`, `pandas`, `numpy`, `pyyaml`, `tenacity`, `tqdm`, `requests`.

## API Keys

The framework never stores API keys in code or config; keys are read from environment variables named by `api_key_env` in `configs/models.yaml` and `embedding.api_key_env` in `configs/rag.yaml`.

```bash
# All default models run through DashScope, so a single key is enough:
export DASHSCOPE_API_KEY="sk-xxxx"

# If you add models from other vendors, set their keys too, e.g.:
export DEEPSEEK_API_KEY="sk-xxxx"
export ZHIPU_API_KEY="sk-xxxx"
```

On Windows PowerShell: `$env:DASHSCOPE_API_KEY = "sk-xxxx"`.

The runner skips any model whose key is missing and reports it in the log instead of aborting. The key is never written to logs.

## Input Data

Place a `data/llm_prompt_cases.jsonl` file (one JSON object per line), or point `--data_path` at your own file. A minimal schema example lives at `data/llm_prompt_cases.sample.jsonl`:

```json
{
  "case_id": 0,
  "gt_vars": ["V27"],
  "top10_vars": ["V27", "V19", "V22", "V31", "V12", "V5", "V8", "V33", "V14", "V2"],
  "top10_details": [
    {"var": "V27", "type": "flow", "score": 0.92, "raw": 12.3, "recon": 11.8, "residual": 0.5, "semantic": "Flow sensor abnormal increase"}
  ],
  "prompt": "optional pre-built full prompt; used as-is if present"
}
```

- `gt_vars` — true root-cause variables (string array or comma-separated string). Used only for offline evaluation, never inserted into the prompt.
- `top10_vars` — candidate Top-10 variable IDs.
- `top10_details` — per-candidate details (`var`, `type`, `score`, `raw`, `recon`, `residual`, `semantic`, …).
- `prompt` (optional) — skip template assembly and use this prompt verbatim.

`gt_vars`/`top10_vars` accept either arrays or comma-separated strings and are normalized automatically.

## Usage

### 1. Run a single model

```bash
python src/main.py \
  --data_path data/llm_prompt_cases.jsonl \
  --config_path configs/models.yaml \
  --model_name qwen-plus \
  --output_dir outputs \
  --num_runs 3 \
  --max_cases 20
```

Common options (defaults in parentheses):

| Argument | Description | Default |
| --- | --- | --- |
| `--model_name` | Model name, must match a key in `models.yaml` | required |
| `--num_runs` | Number of repeated inferences per case | 3 |
| `--max_cases` | Max cases to run (`-1` = all) | -1 |
| `--case_ids` | Run only the given case IDs, preserving order | None |
| `--temperature` | Override sampling temperature | from YAML |
| `--max_tokens` | Override max output tokens | from YAML |
| `--thinking_budget` | Override reasoning-token budget (`0` disables the field) | from YAML |
| `--resume` | Skip already-completed `(run_id, case_id)` combos | False |
| `--resume_skip_failed` | Also skip previously failed cases on resume | 0 |
| `--sleep_ms` | Sleep between API calls (rate limiting) | 300 |
| `--max_retries` | Max retries per API call | 5 |
| `--request_timeout` | Request timeout (seconds) | 120 |
| `--retry_base_sleep` | Base sleep for exponential backoff (seconds) | 5 |
| `--use_rag` | Enable RAG retrieval (`1`/`0`) | 0 |
| `--rag_config` | RAG config path | `configs/rag.yaml` |
| `--rag_top_k` | Override `top_k` in the RAG config | from `rag.yaml` |
| `--use_iterative_agent` | Enable the iterative self-refinement agent (`1`/`0`) | 0 |
| `--use_dual_branch_fusion` | Enable evidence + RAG dual-branch deterministic fusion (`1`/`0`) | 0 |
| `--max_iterations` | Max agent iterations | 3 |
| `--min_confidence` | Validation confidence threshold for the agent | 0.5 |
| `--compact_evidence` | Use compact evidence format to shorten prompts (`1`/`0`) | 0 |

#### API stability guarantees

- Network errors (`ConnectionResetError`, `TimeoutError`, `socket.timeout`, `ssl.SSLError`, `urllib.error.URLError`, `urllib.error.HTTPError`, …) are caught centrally so a single failure never crashes the run.
- Exponential backoff with jitter: `base * 2^(attempt-1)` seconds plus 0–30% jitter.
- After all retries fail, a structured error record is written and the loop continues.
- Every case result is appended to `raw_responses/*.jsonl` and flushed immediately, so an interrupted run loses no data.
- `--resume` scans existing `raw_responses/{model}_run_*.jsonl` and skips completed combos.

Logs are written to `logs/` as `rca_experiment_{model}_{rag_flag}_{timestamp}.log`, where `rag_flag` is `with-rag` or `no-rag`.

### 2. Batch all models

```bash
python src/run_all_models.py \
  --data_path data/llm_prompt_cases.jsonl \
  --config_path configs/models.yaml \
  --output_dir outputs \
  --num_runs 3
```

Run a subset: `python src/run_all_models.py --models qwen-plus glm-5.2 --num_runs 1`.

This sequentially runs each configured model, aggregates metrics, and writes `outputs/metrics/all_models_summary.csv` and `all_models_summary.json`.

### 3. Official ablation matrix (recommended)

`scripts/run_final_experiments.py` runs the paper's ablation matrix serially and validates each group before moving on.

Conditions (default = the first four):

| Condition | RAG | Agent | Dual-branch fusion |
| --- | --- | --- | --- |
| `baseline` | ✗ | ✗ | ✗ |
| `only_rag` | ✓ | ✗ | ✗ |
| `only_self_refinement_agent` | ✗ | ✓ | ✗ |
| `rag_self_refinement_agent` | ✓ | ✓ | ✗ |
| `dual_branch_fusion_agent` | ✓ | ✓ | ✓ |

Examples:

```bash
# All 5 models under the 4 default conditions
python scripts/run_final_experiments.py --num-runs 3 --max-cases 20 --max-tokens 8192

# Selected models and conditions
python scripts/run_final_experiments.py \
  --models qwen-plus qwen-max \
  --conditions baseline only_rag \
  --num-runs 3 --max-cases 20

# Preflight on known worst-case cases into a separate directory (no pollution)
python scripts/run_final_experiments.py \
  --models deepseek-v4-flash deepseek-v4-pro glm-5.2 \
  --conditions baseline \
  --num-runs 1 --case-ids 12 13 16 --max-tokens 8192 \
  --output-root outputs/thinking_budget_preflight

# Print commands only, without calling any API
python scripts/run_final_experiments.py --dry-run
```

Options: `--models`, `--conditions`, `--num-runs`, `--max-cases`, `--case-ids`, `--temperature` (default 0.2), `--max-tokens` (default 8192), `--resume`, `--dry-run`, `--output-root`. Each group is written under `outputs/{condition}_{model}/` with `raw_responses/`, `parsed_results/`, `metrics/`, and `logs/` subdirectories.

### 4. Build the RAG index

```bash
python src/build_rag_index.py \
  --data_dir data/swat_s2s_raw_window_fixed \
  --kb_dir data/rag_kb \
  --rag_config configs/rag.yaml \
  --rebuild
```

The vectors are persisted to `outputs/chroma_db/`. (The SWaT window dataset under `data/swat_s2s_raw_window_fixed` is an external dataset and is not committed; the knowledge-base markdown under `data/rag_kb/` is committed.)

### 5. Inspect retrieval

```bash
python src/test_rag_retrieval.py \
  --data_path data/swat_s2s_raw_window_fixed/llm_prompt_cases.jsonl \
  --rag_config configs/rag.yaml \
  --case_id 0
```

## Methods: RAG, Iterative Self-refinement Agent, and Dual-Branch Fusion

- **RAG** — an optional lightweight ChromaDB + Embedding-API pipeline injects top-k general industrial knowledge into the prompt. Embedded content contains only generic background (variable meaning, SWaT P1–P6 process, sensor/actuator relations, common fault modes). Ground-truth answers and case-level labels are never embedded, preventing data leakage. If no embedding key is configured, or retrieval fails, the pipeline logs a warning and silently falls back to a no-RAG prompt.
- **Iterative Self-refinement Agent** (`src/iterative_agent.py`) — round 1 does initial RCA reasoning; then the response is validated against structural and content rules. On failure, round 2 refines based on the specific reasons; if it still fails, round 3 makes a forced-choice selection under strong constraints. A validated result is returned, otherwise `UNKNOWN`. Ground truth is never accessed during agent operation.
- **Dual-Branch Fusion** (`src/dual_branch_fusion_agent.py`) — runs the data-evidence agent branch and the RAG knowledge branch independently and fuses them deterministically at the candidate level: the evidence branch decides the primary cause (so generic knowledge cannot override on-site time series), and the RAG branch only supplements candidates 2–5.
- **Response Validator** (`src/response_validator.py`) — verifies that output is valid JSON, all variables come from the Top-10 candidates, confidence is in-range, and required explanation fields are non-empty.

## Outputs and Metrics

- `outputs/raw_responses/{model}_run_{n}.jsonl` — full audit trail: prompt, raw output, parse result, GT, Top-10, hit flags, and agent/retrieval fields.
- `outputs/parsed_results/{model}_parsed.csv` — flattened per-record results.
- `outputs/metrics/{model}_metrics.json` — per-model aggregated metrics.
- `outputs/metrics/all_models_summary.csv` / `.json` — cross-model comparison.

Core metrics:

| Metric | Definition |
| --- | --- |
| `accuracy_at_1/3/5` | Fraction of records where the true root cause is within the top-1/3/5 predictions |
| `covered_accuracy_at_1/3/5` | Hit@K computed only on records whose true cause is already inside the Top-10 (isolates LLM re-ranking ability) |
| `valid_json_rate` | Fraction of final results parseable as valid JSON |
| `hallucination_rate` | Fraction of outputs containing variables outside the Top-10 (lower is better) |
| `average_confidence` | Mean model-reported confidence (for analysis only, not accuracy) |
| `average_iteration_count` / Agent metrics | Iterative-agent rounds used, per-round recovery rates, and final failure rate |
| `api_success_rate`, `truncated_response_count/rate`, token usage | Run quality and configuration diagnostics |

## Testing and Utilities

- `scripts/make_sample_data.py` — writes a small synthetic `llm_prompt_cases.jsonl` for smoke tests.
- `scripts/smoke_test*.py` — end-to-end smoke tests with a mocked LLM (no real API cost).
- `scripts/verify_csv.py` — checks that every row of a `{model}_parsed.csv` has the required fields populated.
- `python -m pytest src` — runs the unit tests in `src/test_*.py`.

## Additional Experiment Modules (`experiments/`)

- `mechanism_validation_v1/` — three mechanism checks without touching the main SWaT experiment or stored branch outputs.
- `react_adapted_v1/` — a ReAct-style tool-based agent adapted to frozen evidence, the domain knowledge base, and candidate-ranked submissions.
- `reproducibility_addendum_v1/` — exports KDAgent's reproducibility materials from existing results.
- `wadi_external_validation_v1/` — fixed-protocol external validation on the WADI dataset (baseline vs. KDAgent authority-preserving), with its own requirements (`requirements-wadi.txt`).

Each module has its own `README.md`.

## Offline Analysis (`analysis/`)

Read-only scripts that recompute metrics and audit behavior from already-saved results **without** calling any LLM API:

- `build_swat_episode_audit.py` — restores auditable per-episode timing from the raw SWaT attack workbook.
- `preflight_prompt_alignment.py` — offline audit and run entry for prompt-alignment v2.
- `run_rrf_borda_posthoc.py` — deterministic RRF/Borda fusion over saved clean and retrieval-robustness results.
- `run_strong_baseline_reevaluation.py` — unifies metric recomputation and paired comparison tables across methods.

## Configuration

`configs/models.yaml` — list models under `models:`. Each entry: `name`, `provider` (`openai_compatible`), `base_url`, `api_key_env`, `model`, and optional `temperature`, `max_tokens`, `thinking_budget`, `timeout`, `max_retries`, `retry_base_sleep`. Defaults ship with `qwen-plus`, `qwen-max`, `deepseek-v4-flash`, `deepseek-v4-pro`, and `glm-5.2` (temperature 0.2, `max_tokens` 8192, `thinking_budget` 2048 for the DeepSeek/GLM reasoning models).

`configs/rag.yaml` — embedding provider (OpenAI-compatible, `text-embedding-v4`), vector-store persistence (`outputs/chroma_db`, collection `swat_process_kb`), chunking, and retrieval (`top_k` 5).

## FAQ

1. **Model name not recognized** — `--model_name` must match a `name` in `models.yaml` (matched case-insensitively).
2. **API key missing** — set `DASHSCOPE_API_KEY` (and any other keys referenced by `api_key_env`); affected models are skipped with a log note.
3. **Rate limiting / timeouts** — increase `--sleep_ms` or `--max_retries`.
4. **Add a new model** — append an entry under `models:` in `configs/models.yaml`; no code changes needed.
5. **Customize the prompt** — edit `prompts/rca_prompt_template.txt`, keeping the `{process_knowledge}` and `{top10_candidates}` placeholders.