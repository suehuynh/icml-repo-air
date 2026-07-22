"""
Loads the two trained checkpoints (baseline GRPO, GRPO+AIR) and runs both
evaluators against ID and OOD splits, printing logbook-ready reports.

    python -m eval.run_evaluation --log-level INFO

Run this AFTER both training.grpo_baseline and training.train_air have
completed (real runs, not just smoke tests, smoke-test checkpoints only
had 2 steps and won't show a meaningful signal).
"""

from __future__ import annotations

import argparse
import logging

from config import CONFIG
from data.advbench_loader import load_advbench_intents, build_meta_groups
from data.group_sampler import HeterogeneousGroupSampler
from eval.anchor_accuracy import run_id_evaluation, AnchorAccuracyEvaluator
from eval.ood_consistency import run_ood_evaluation, OODConsistencyEvaluator

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # deferred-heavy
    import torch

    has_cuda = torch.cuda.is_available()
    # token=True: use whatever's cached from `hf auth login` (or HF_TOKEN
    # in the environment, e.g. on an HF GPU Job with --secrets HF_TOKEN).
    # Required since the training scripts push to PRIVATE repos
    # (CONFIG.hub.hub_private=True), a public repo wouldn't need this.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_id,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
        token=True,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-checkpoint",
        default=CONFIG.hub.baseline_repo_id,
        help="Local path or Hub repo ID (defaults to the Hub repo the baseline run pushed to).",
    )
    parser.add_argument(
        "--air-checkpoint",
        default=CONFIG.hub.air_repo_id,
        help="Local path or Hub repo ID (defaults to the Hub repo the AIR run pushed to).",
    )
    parser.add_argument(
        "--max-id-groups",
        type=int,
        default=15,
        help=(
            "CRITICAL for cost control: the real training run used all ~520 "
            "AdvBench behaviors (~416 ID groups). Evaluating that many "
            "groups x 2 checkpoints x up to 11 judge calls per open prompt "
            "would cost far more than a typical remaining GPU budget. This "
            "caps the ID split to a small, honest SAMPLE instead. Document "
            "this sampling in your logbook, results are from a subset, not "
            "the full ID set."
        ),
    )
    parser.add_argument(
        "--max-ood-groups",
        type=int,
        default=15,
        help="Same cost-control reasoning as --max-id-groups, applied to the OOD split (~104 groups full).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation length per prompt during eval. Matches training's max_completion_length by default.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    intents = load_advbench_intents()
    id_groups, ood_groups = build_meta_groups(intents)

    logger.info(
        "Full dataset: %d ID groups, %d OOD groups. Sampling down to %d / %d for cost control.",
        len(id_groups), len(ood_groups), args.max_id_groups, args.max_ood_groups,
    )
    id_groups = id_groups[: args.max_id_groups]
    ood_groups = ood_groups[: args.max_ood_groups]

    id_dataset = HeterogeneousGroupSampler(id_groups)
    ood_dataset = HeterogeneousGroupSampler(ood_groups)

    logger.info("Loading baseline checkpoint: %s", args.baseline_checkpoint)
    baseline_model, baseline_tokenizer = load_model_and_tokenizer(args.baseline_checkpoint)

    logger.info("Evaluating baseline on ID split...")
    baseline_id_summary = run_id_evaluation(
        baseline_model, baseline_tokenizer, id_dataset, max_new_tokens=args.max_new_tokens
    )
    logger.info("Evaluating baseline on OOD split...")
    baseline_ood_summary = run_ood_evaluation(
        baseline_model, baseline_tokenizer, ood_dataset, max_new_tokens=args.max_new_tokens
    )

    del baseline_model  # free memory before loading the second checkpoint
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Loading AIR checkpoint: %s", args.air_checkpoint)
    air_model, air_tokenizer = load_model_and_tokenizer(args.air_checkpoint)

    logger.info("Evaluating AIR on ID split...")
    air_id_summary = run_id_evaluation(
        air_model, air_tokenizer, id_dataset, max_new_tokens=args.max_new_tokens
    )
    logger.info("Evaluating AIR on OOD split...")
    air_ood_summary = run_ood_evaluation(
        air_model, air_tokenizer, ood_dataset, max_new_tokens=args.max_new_tokens
    )

    id_report = AnchorAccuracyEvaluator().report(baseline_id_summary, air_id_summary)
    ood_report = OODConsistencyEvaluator().report(baseline_ood_summary, air_ood_summary)

    print("\n" + "=" * 70)
    print(id_report)
    print("=" * 70)
    print(ood_report)
    print("=" * 70)
    print("\nCopy the two report blocks above directly into your Trackio logbook.")

    # BACKSTOP, added given how much trust this project has lost to "should
    # be fine" assumptions about persistence. Job logs (the print()s above)
    # should survive after the container is torn down, same as we observed
    # for the training runs' log output, but this pushes a small durable
    # copy to the Hub too, costing next to nothing (a few KB of text)
    # rather than relying solely on log retention.
    try:
        from huggingface_hub import HfApi
        import os

        full_report = (
            f"Evaluation run: {args.max_id_groups} ID groups, {args.max_ood_groups} OOD groups sampled\n"
            f"Baseline checkpoint: {args.baseline_checkpoint}\n"
            f"AIR checkpoint: {args.air_checkpoint}\n\n"
            f"{id_report}\n\n{ood_report}\n"
        )
        report_path = "eval_report.txt"
        with open(report_path, "w") as f:
            f.write(full_report)

        api = HfApi(token=os.environ.get("HF_TOKEN") or True)
        api.upload_file(
            path_or_fileobj=report_path,
            path_in_repo="eval_report.txt",
            repo_id=CONFIG.hub.air_repo_id,
            repo_type="model",
        )
        logger.info(
            "Backstop copy pushed: https://huggingface.co/%s/blob/main/eval_report.txt",
            CONFIG.hub.air_repo_id,
        )
    except Exception as e:
        logger.warning(
            "Could not push backstop report copy to the Hub (%s). The printed "
            "report above should still be readable via `hf jobs logs`, copy "
            "it manually now just in case.",
            e,
        )


if __name__ == "__main__":
    main()