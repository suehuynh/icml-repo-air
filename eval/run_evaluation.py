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


def load_model_and_tokenizer(checkpoint_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # deferred-heavy
    import torch

    has_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", default=CONFIG.output_dir)
    parser.add_argument("--air-checkpoint", default=CONFIG.output_dir + "_air")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    intents = load_advbench_intents()
    id_groups, ood_groups = build_meta_groups(intents)
    id_dataset = HeterogeneousGroupSampler(id_groups)
    ood_dataset = HeterogeneousGroupSampler(ood_groups)

    logger.info("Loading baseline checkpoint: %s", args.baseline_checkpoint)
    baseline_model, baseline_tokenizer = load_model_and_tokenizer(args.baseline_checkpoint)

    logger.info("Evaluating baseline on ID split...")
    baseline_id_summary = run_id_evaluation(baseline_model, baseline_tokenizer, id_dataset)
    logger.info("Evaluating baseline on OOD split...")
    baseline_ood_summary = run_ood_evaluation(baseline_model, baseline_tokenizer, ood_dataset)

    del baseline_model  # free memory before loading the second checkpoint
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Loading AIR checkpoint: %s", args.air_checkpoint)
    air_model, air_tokenizer = load_model_and_tokenizer(args.air_checkpoint)

    logger.info("Evaluating AIR on ID split...")
    air_id_summary = run_id_evaluation(air_model, air_tokenizer, id_dataset)
    logger.info("Evaluating AIR on OOD split...")
    air_ood_summary = run_ood_evaluation(air_model, air_tokenizer, ood_dataset)

    print("\n" + "=" * 70)
    print(AnchorAccuracyEvaluator().report(baseline_id_summary, air_id_summary))
    print("=" * 70)
    print(OODConsistencyEvaluator().report(baseline_ood_summary, air_ood_summary))
    print("=" * 70)
    print("\nCopy the two report blocks above directly into your Trackio logbook.")


if __name__ == "__main__":
    main()