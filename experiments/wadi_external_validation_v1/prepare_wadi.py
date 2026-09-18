"""Train and freeze the WADI TA-RCA, exporting candidate evidence compatible with the existing KDAgent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from experiments.wadi_external_validation_v1.wadi_adapter import (
    build_episode_manifest,
    enrich_catalog_types,
    locate_wadi_files,
    read_numeric_process_data,
    sampled_context,
    select_prompt_indices,
    split_semicolon,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "wadi"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "wadi_external_validation_v1"
DEFAULT_AERCA_SOURCE = Path(r"D:\workspace\python\AERCA_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 WADI 外部验证的冻结 TA-RCA 证据。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aerca-source", type=Path, default=DEFAULT_AERCA_SOURCE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--chunk-len", type=int, default=2000)
    parser.add_argument("--context-radius", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wadi_prepare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_dir / "prepare_wadi.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _import_aerca(aerca_source: Path):
    """Defer importing deep-learning dependencies so that the manifest audit does not depend on PyTorch."""
    if not aerca_source.exists():
        raise FileNotFoundError(f"AERCA 源码目录不存在：{aerca_source}")
    source_text = str(aerca_source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        import torch
        from models.aerca import AERCA
        from utils.preprocessor import AERCAPreprocessor
        from utils.utils import set_seed
    except ImportError as exc:
        raise RuntimeError(
            "WADI TA-RCA 需要 torch、scipy、scikit-learn 和 tqdm。"
            "请先安装 experiments/wadi_external_validation_v1/requirements-wadi.txt。"
        ) from exc
    return torch, AERCA, AERCAPreprocessor, set_seed


def _write_manifest(output_dir: Path, manifest: pd.DataFrame, catalog: List[Dict[str, Any]]) -> None:
    data_dir = output_dir / "prepared_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = json.loads(manifest.to_json(orient="records", force_ascii=False))
    for destination in (data_dir, output_dir):
        manifest.to_csv(destination / "wadi_episode_manifest.csv", index=False, encoding="utf-8")
        write_json(destination / "wadi_episode_manifest.json", manifest_records)
    write_json(data_dir / "variable_catalog.json", catalog)


def _training_config(args: argparse.Namespace, num_vars: int) -> Dict[str, Any]:
    """Reuse the current WADI AERCA parameters as-is, without tuning them to test results."""
    return {
        "num_vars": num_vars,
        "hidden_layer_size": 16,
        "num_hidden_layers": 1,
        "window_size": 1,
        "stride": 1,
        "encoder_alpha": 0.5,
        "decoder_alpha": 0.5,
        "encoder_gamma": 0.5,
        "decoder_gamma": 0.5,
        "encoder_lambda": 0.5,
        "decoder_lambda": 0.5,
        "beta": 0.5,
        "lr": 1e-4,
        "epochs": args.epochs,
        "recon_threshold": 0.95,
        "causal_quantile": 0.80,
        "root_cause_threshold_encoder": 0.95,
        "root_cause_threshold_decoder": 0.95,
        "risk": 1e-2,
        "initial_level": 0.98,
        "num_candidates": 100,
        "seed": args.seed,
        "downsample": args.downsample,
        "chunk_len": args.chunk_len,
        "context_radius": args.context_radius,
        "score_fusion": "0.98*encoder_z+0.02*decoder_z",
        "center_strategy": "first_valid_attack_point",
        "case_score": "mean_post_center_within_fixed_context",
    }


def _create_model(AERCA, config: Dict[str, Any], device: str):
    return AERCA(
        num_vars=config["num_vars"],
        hidden_layer_size=config["hidden_layer_size"],
        num_hidden_layers=config["num_hidden_layers"],
        device=device,
        window_size=config["window_size"],
        stride=config["stride"],
        encoder_alpha=config["encoder_alpha"],
        decoder_alpha=config["decoder_alpha"],
        encoder_gamma=config["encoder_gamma"],
        decoder_gamma=config["decoder_gamma"],
        encoder_lambda=config["encoder_lambda"],
        decoder_lambda=config["decoder_lambda"],
        beta=config["beta"],
        lr=config["lr"],
        epochs=config["epochs"],
        recon_threshold=config["recon_threshold"],
        data_name="wadi_external_v1",
        causal_quantile=config["causal_quantile"],
        root_cause_threshold_encoder=config["root_cause_threshold_encoder"],
        root_cause_threshold_decoder=config["root_cause_threshold_decoder"],
        risk=config["risk"],
        initial_level=config["initial_level"],
        num_candidates=config["num_candidates"],
    )


def _checkpoint_paths(model, checkpoint_dir: Path) -> List[Path]:
    prefix = checkpoint_dir / model.model_name
    suffixes = (
        ".pt",
        "_recon_threshold.npy",
        "_recon_mean.npy",
        "_recon_std.npy",
        "_lower_encoder.npy",
        "_upper_encoder.npy",
        "_us_mean_encoder.npy",
        "_us_std_encoder.npy",
        "_lower_decoder.npy",
        "_upper_decoder.npy",
        "_us_mean_decoder.npy",
        "_us_std_decoder.npy",
    )
    return [Path(str(prefix) + suffix) for suffix in suffixes]


def _fit_preprocessor(AERCAPreprocessor, normal_chunks: np.ndarray, attack_data: np.ndarray):
    preprocessor = AERCAPreprocessor(
        n_regimes=3,
        regime_feature_topk=12,
        blend_alpha=0.7,
        clip_value=5.0,
        min_regime_samples=8,
        random_state=42,
    )
    normal_processed = preprocessor.fit_transform(normal_chunks)
    attack_processed = preprocessor.transform(attack_data[None, :, :])[0]
    return preprocessor, normal_processed, attack_processed


def _save_preprocessor(output_dir: Path, preprocessor) -> None:
    np.savez_compressed(
        output_dir / "prepared_data" / "wadi_preprocessor.npz",
        discrete_mask=preprocessor.discrete_mask,
        global_median=preprocessor.global_median,
        global_iqr=preprocessor.global_iqr,
        clip_value=np.asarray([preprocessor.clip_value]),
    )


def _load_model_state(torch, model, checkpoint_dir: Path) -> None:
    prefix = checkpoint_dir / model.model_name
    model.load_state_dict(torch.load(str(prefix) + ".pt", map_location=model.device))
    model.us_mean_encoder = np.load(str(prefix) + "_us_mean_encoder.npy")
    model.us_std_encoder = np.load(str(prefix) + "_us_std_encoder.npy")
    model.us_mean_decoder = np.load(str(prefix) + "_us_mean_decoder.npy")
    model.us_std_decoder = np.load(str(prefix) + "_us_std_decoder.npy")
    model.eval()


def _train_with_fixed_update_budget(torch, model, normal_processed: np.ndarray, update_budget: int, logger: logging.Logger) -> None:
    """Train with the update-budget semantics of the original single-sequence AERCA, so that chunking does not amplify the budget by 62x.

The original `_training` performs one optimizer step per sequence chunk per epoch. After WADI
is split into multiple chunks to avoid OOM, 5000 epochs would accidentally turn into hundreds
of thousands of updates. This fixes the total number of updates to 5000, cycling through all
training chunks in order; the 80/20 split, validation objective, and patience=20 are kept as-is.
"""
    from tqdm import tqdm

    split = int(0.8 * len(normal_processed))
    train_data = normal_processed[:split]
    validation_data = normal_processed[split:]
    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("TA-RCA 训练/验证分块为空。")

    best_validation = np.inf
    bad_validation_rounds = 0
    checkpoint_path = Path(model.save_dir) / f"{model.model_name}.pt"
    completed_updates = 0
    for update in tqdm(range(update_budget), desc="TA-RCA updates"):
        model.train()
        model.optimizer.zero_grad()
        loss = model._training_step(train_data[update % len(train_data)])
        loss.backward()
        model.optimizer.step()
        completed_updates = update + 1

        # Validate only once per full pass over the training chunks, preserving the "one full data pass" early-stopping semantics.
        end_of_pass = completed_updates % len(train_data) == 0
        if not end_of_pass and completed_updates != update_budget:
            continue
        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for chunk in validation_data:
                validation_loss += float(model._training_step(chunk).item())
        if np.isfinite(validation_loss) and validation_loss < best_validation:
            best_validation = validation_loss
            bad_validation_rounds = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_validation_rounds += 1
        logger.info(
            "TA-RCA updates=%d/%d, validation_loss=%.6f, patience=%d/20。",
            completed_updates,
            update_budget,
            validation_loss,
            bad_validation_rounds,
        )
        if bad_validation_rounds >= 20:
            logger.info("TA-RCA early stopping：连续 20 次全量验证未改善。")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("TA-RCA 训练结束但未产生有效 checkpoint。")
    model.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
    model._get_recon_threshold(validation_data)
    model._get_root_cause_threshold_encoder(validation_data)
    model._get_root_cause_threshold_decoder(validation_data)
    logger.info("TA-RCA 实际 optimizer updates=%d。", completed_updates)


def _trend_description(values: np.ndarray, metadata: Dict[str, Any]) -> str:
    start = float(values[0])
    end = float(values[-1])
    scale = max(float(np.ptp(values)), 1e-6)
    relative_change = abs(end - start) / scale
    if relative_change < 0.1:
        trend = "approximately stable over the exported window"
    elif end > start:
        trend = "increases over the exported window"
    else:
        trend = "decreases over the exported window"
    return f"{metadata['description']}; observed signal {trend}"


def _export_case(
    torch,
    model,
    episode: pd.Series,
    catalog: List[Dict[str, Any]],
    raw_attack: np.ndarray,
    processed_attack: np.ndarray,
    sampled_rows: np.ndarray,
    context_radius: int,
) -> Dict[str, Any]:
    start, end, center_local = sampled_context(episode, sampled_rows, context_radius)
    raw_context = raw_attack[start : end + 1]
    model_context = processed_attack[start : end + 1]
    if len(model_context) <= model.window_size * 2:
        raise ValueError(f"{episode['episode_id']} 的上下文过短：{len(model_context)}")

    with torch.no_grad():
        _, reconstructed, targets, _, _, _, _, encoder_residual = model._testing_step(
            model_context, add_u=False
        )
    enc = encoder_residual[model.window_size :].cpu().numpy()
    dec = (targets - reconstructed).cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    targets_np = targets.cpu().numpy()
    enc_score = np.abs(enc - model.us_mean_encoder) / np.maximum(model.us_std_encoder, 1e-6)
    dec_score = np.abs(dec - model.us_mean_decoder) / np.maximum(model.us_std_decoder, 1e-6)
    final_score = 0.98 * enc_score + 0.02 * dec_score

    # The model's first alignable output corresponds to the original context index 2; candidate scores strictly reuse
    # the SWaT exporter's first-valid convention, averaging only the observable scores after the attack starts.
    aligned_local = np.arange(model.window_size * 2, len(model_context))
    post_center = aligned_local >= center_local
    if not post_center.any():
        raise ValueError(f"{episode['episode_id']} 没有攻击开始后的模型输出。")
    case_scores = np.nan_to_num(final_score[post_center].mean(axis=0), nan=0.0)
    order = sorted(
        range(len(catalog)),
        key=lambda index: (-float(case_scores[index]), catalog[index]["var_id"]),
    )[:10]

    # The existing Prompt formatter shows at most 12 time points, so the exporter draws 12 evenly spaced points directly,
    # avoiding an evidence crop limited to the first half of the window and losing post-attack evidence.
    prompt_indices = select_prompt_indices(len(model_context), maximum=12)
    top_k: List[Dict[str, Any]] = []
    aligned_lookup = {int(local): pos for pos, local in enumerate(aligned_local)}
    for rank, var_index in enumerate(order, start=1):
        metadata = catalog[var_index]
        time_series: List[Dict[str, Any]] = []
        for local in prompt_indices:
            aligned_pos = aligned_lookup.get(local)
            item: Dict[str, Any] = {
                "time_offset": int(local - center_local),
                "source_row": int(sampled_rows[start + local]),
                "raw_value": float(raw_context[local, var_index]),
                "model_input_value": float(model_context[local, var_index]),
                "recon_value": None,
                "target_model_value": None,
                "encoder_residual": None,
                "decoder_residual": None,
                "encoder_score": None,
                "decoder_score": None,
                "final_score": None,
            }
            if aligned_pos is not None:
                item.update(
                    {
                        "recon_value": float(reconstructed_np[aligned_pos, var_index]),
                        "target_model_value": float(targets_np[aligned_pos, var_index]),
                        "encoder_residual": float(enc[aligned_pos, var_index]),
                        "decoder_residual": float(dec[aligned_pos, var_index]),
                        "encoder_score": float(enc_score[aligned_pos, var_index]),
                        "decoder_score": float(dec_score[aligned_pos, var_index]),
                        "final_score": float(final_score[aligned_pos, var_index]),
                    }
                )
            time_series.append(item)
        top_k.append(
            {
                "rank": rank,
                "var_id": metadata["var_id"],
                "var_index": var_index,
                "name": metadata["original_name"],
                "type": metadata["var_type"],
                "stage": metadata["stage"],
                "role": metadata["role"],
                "case_level_score": float(case_scores[var_index]),
                "description": _trend_description(raw_context[:, var_index], metadata),
                "time_series": time_series,
            }
        )

    gt_vars = split_semicolon(episode["gt_vars"])
    top10_vars = [item["var_id"] for item in top_k]
    best_rank = next((index + 1 for index, value in enumerate(top10_vars) if value in gt_vars), None)
    return {
        "case_id": episode["case_id"],
        "episode_id": episode["episode_id"],
        "system_name": "WADI",
        "task": "WADI water-distribution root cause analysis",
        "source_attack_id": episode["source_attack_id"],
        "attack_interval": {
            "row_start": int(episode["attack_row_start"]),
            "row_end": int(episode["attack_row_end"]),
            "declared_start_time": episode["declared_start_time"],
            "declared_end_time": episode["declared_end_time"],
        },
        "evidence_window": {
            "sampled_position_start": start,
            "sampled_position_end": end,
            "center_local_index": center_local,
            "context_radius": context_radius,
            "context_length": len(model_context),
            "center_strategy": "first_valid_attack_point",
            "downsample": int(sampled_rows[1] - sampled_rows[0]),
        },
        "top_k_variables": top_k,
        "ground_truth": {
            "gt_vars": gt_vars,
            "gt_names": split_semicolon(episode["gt_names"]),
            "best_gt_rank_in_numeric_rca": best_rank,
        },
    }


def _write_kb_inputs(output_dir: Path, catalog: List[Dict[str, Any]]) -> None:
    data_dir = output_dir / "prepared_data"
    kb_dir = data_dir / "rag_kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog).to_csv(data_dir / "variables_meta.csv", index=False, encoding="utf-8")

    process_text = """# WADI process knowledge

WADI is a water-distribution testbed containing source-water instrumentation, tanks,
pumps, motorized or modulating valves, pressure/flow/level sensors, and consumer
distribution branches. Prefix `1_` and `2_` identify the two recorded subsystems.

## Generic diagnostic rules

Actuator or setpoint changes can precede downstream flow, pressure, and level responses.
Sensor anomalies without a physically consistent actuator or upstream response may be
measurement faults. Root-cause ranking must combine temporal precedence, persistent
residuals, signal type, and physically plausible propagation. These statements are
generic system knowledge and contain no attack labels or episode answers.
"""
    (data_dir / "process_knowledge_template.md").write_text(process_text, encoding="utf-8")
    relation_text = """# Generic WADI sensor-actuator relations

- A motorized or modulating valve controls a hydraulic path; its abnormal command can alter downstream flow and pressure.
- Pump state changes can alter branch flow, tank level, and downstream pressure.
- A tank level is governed by the balance between inlet and outlet flows.
- Pressure-control setpoints can propagate to controller output, valve position, flow, and measured pressure.
- Analyzer readings describe water quality and should be interpreted together with upstream flow paths and treatment states.
- Alarm and switch variables are supporting state evidence; they do not by themselves prove the initiating component.
"""
    (kb_dir / "wadi_generic_relations.md").write_text(relation_text, encoding="utf-8")


def _assert_kb_has_no_case_answers(output_dir: Path, manifest: pd.DataFrame) -> None:
    """Prevent episode identifiers, attack notes, or answer fields from entering the retrieval corpus."""
    data_dir = output_dir / "prepared_data"
    texts = [
        (data_dir / "process_knowledge_template.md").read_text(encoding="utf-8"),
        *[path.read_text(encoding="utf-8") for path in (data_dir / "rag_kb").glob("*.md")],
    ]
    corpus = "\n".join(texts).lower()
    forbidden = ["ground_truth", "gt_vars", "root_tags"]
    forbidden.extend(str(value).lower() for value in manifest["episode_id"])
    leaked = [token for token in forbidden if token and token in corpus]
    if leaked:
        raise ValueError(f"WADI KB 泄漏 case/答案字段：{leaked}")


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    if args.downsample <= 0 or args.chunk_len <= 2 or args.context_radius <= 0:
        raise ValueError("downsample、chunk-len、context-radius 必须为正数。")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    files = locate_wadi_files(args.data_dir.resolve())
    manifest, catalog = build_episode_manifest(files)
    _write_manifest(output_dir, manifest, catalog)
    logger.info(
        "Episode 审计完成：连续攻击区间=%d，可评估=%d，过程变量=%d。",
        len(manifest),
        int(manifest["included"].sum()),
        len(catalog),
    )
    if args.manifest_only:
        return {"episodes": len(manifest), "included": int(manifest["included"].sum())}

    torch, AERCA, AERCAPreprocessor, set_seed = _import_aerca(args.aerca_source.resolve())
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 Python 环境中的 PyTorch 无法访问 GPU。")

    feature_names = [item["original_name"] for item in catalog]
    logger.info("读取并清洗 WADI normal data...")
    normal = read_numeric_process_data(files.normal, 0, feature_names)[:: args.downsample]
    logger.info("读取并清洗 WADI attack data...")
    attack = read_numeric_process_data(files.attack, 1, feature_names)[:: args.downsample]
    sampled_rows = np.arange(0, len(pd.read_csv(files.attack, header=1, usecols=[0])), args.downsample)
    if len(sampled_rows) != len(attack):
        raise ValueError("下采样来源行索引与攻击数据长度不一致。")

    enrich_catalog_types(catalog, normal)
    _write_manifest(output_dir, manifest, catalog)
    _write_kb_inputs(output_dir, catalog)
    _assert_kb_has_no_case_answers(output_dir, manifest)

    usable = (len(normal) // args.chunk_len) * args.chunk_len
    if usable == 0:
        raise ValueError("正常数据不足一个训练 chunk。")
    normal_chunks = normal[:usable].reshape(-1, args.chunk_len, len(catalog))
    logger.info("预处理：normal_chunks=%s, attack=%s。", normal_chunks.shape, attack.shape)
    preprocessor, normal_processed, attack_processed = _fit_preprocessor(
        AERCAPreprocessor, normal_chunks, attack
    )
    _save_preprocessor(output_dir, preprocessor)

    config = _training_config(args, len(catalog))
    config["training_schedule"] = "fixed_total_optimizer_updates_cycling_all_training_chunks"
    config["early_stopping_patience_full_passes"] = 20
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "training_config.json"
    if config_path.exists() and not args.force_train:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("已有 checkpoint 配置与当前命令不同；请恢复原参数或显式 --force-train。")
    write_json(config_path, config)

    set_seed(args.seed)
    model = _create_model(AERCA, config, args.device)
    model.save_dir = str(checkpoint_dir)
    checkpoints = _checkpoint_paths(model, checkpoint_dir)
    checkpoint_complete = all(path.exists() for path in checkpoints)
    if args.force_train or not checkpoint_complete:
        logger.info("开始训练 WADI TA-RCA；optimizer updates=%d，seed=%d。", args.epochs, args.seed)
        _train_with_fixed_update_budget(
            torch,
            model,
            normal_processed,
            update_budget=args.epochs,
            logger=logger,
        )
        logger.info("TA-RCA 训练与阈值估计完成，checkpoint 已冻结。")
    else:
        logger.info("检测到完整冻结 checkpoint，跳过训练。")

    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise RuntimeError(f"TA-RCA checkpoint 不完整：{missing}")
    _load_model_state(torch, model, checkpoint_dir)

    cases = [
        _export_case(
            torch,
            model,
            episode,
            catalog,
            attack,
            attack_processed,
            sampled_rows,
            args.context_radius,
        )
        for _, episode in manifest[manifest["included"]].iterrows()
    ]
    by_id = {case["episode_id"]: case for case in cases}
    for row_index, episode in manifest.iterrows():
        case = by_id.get(episode["episode_id"])
        if case is None:
            continue
        top10 = [item["var_id"] for item in case["top_k_variables"]]
        manifest.at[row_index, "tarca_top10"] = ";".join(top10)
        manifest.at[row_index, "candidate_covered"] = any(
            gt in top10 for gt in case["ground_truth"]["gt_vars"]
        )
        manifest.at[row_index, "best_gt_rank"] = case["ground_truth"]["best_gt_rank_in_numeric_rca"]
    _write_manifest(output_dir, manifest, catalog)

    prepared_dir = output_dir / "prepared_data"
    write_jsonl(prepared_dir / "llm_prompt_cases.jsonl", cases)
    write_json(
        prepared_dir / "wadi_tarca_top10.json",
        {
            "method": "frozen_ta_rca",
            "candidate_count": 10,
            "cases": [
                {
                    "episode_id": case["episode_id"],
                    "ground_truth": case["ground_truth"],
                    "top10": case["top_k_variables"],
                }
                for case in cases
            ],
        },
    )
    # The root-directory copy is the final delivery entry point; the identically named file under prepared_data remains the runtime input.
    write_json(
        output_dir / "wadi_tarca_top10.json",
        json.loads((prepared_dir / "wadi_tarca_top10.json").read_text(encoding="utf-8")),
    )
    coverage = float(manifest.loc[manifest["included"], "candidate_covered"].mean())
    summary = {
        "num_raw_attack_intervals": len(manifest),
        "num_evaluable_episodes": len(cases),
        "num_variables": len(catalog),
        "ta_rca_top10_coverage": coverage,
        "context_radius": args.context_radius,
        "maximum_context_length": args.context_radius * 2 + 1,
        "downsample": args.downsample,
        "ground_truth_policy": "root_tags_only",
        "episode_alignment": "contiguous_attack_labels_to_mapping_fixed_order",
    }
    write_json(prepared_dir / "export_summary.json", summary)
    logger.info("候选证据导出完成：cases=%d，TA-RCA Top-10 coverage=%.4f。", len(cases), coverage)
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    result = prepare(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
