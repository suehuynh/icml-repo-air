"""
Phase 1 baseline: standard GRPO on the Safety heterogeneous meta-groups,
NO AIR regularization yet. This is your first logbook page's result,
the reference point that AIR (Phase 2) must beat on Acc_group and OOD
consistency.

Run shape (smoke test locally first, then move to an HF GPU Job per the
challenge FAQ: "Use a local run to smoke-test code, then run the actual
scaled experiment on a Hugging Face GPU Job and record its URL, GPU type,
command, configuration, and results in your logbook.").

    python -m training.grpo_baseline --smoke-test      # tiny local run
    python -m training.grpo_baseline                   # full config run

TODO before real training (log these as you resolve them):
  - Confirm the reward function signature TRL's GRPOTrainer expects in your
    installed TRL version; the `reward_funcs` interface has changed across
    releases, pin a version and note it in requirements.txt.
  - The AIR auxiliary loss (Phase 2) is NOT implemented here. This script
    is intentionally the GRPO-only baseline. Phase 2 will subclass
    GRPOTrainer as AIRTrainer and override the loss computation using the
    per-group anchor/open reward means this script already logs.
"""

from __future__ import annotations

import argparse
import logging

from config import CONFIG
from data.advbench_loader import load_advbench_intents, build_meta_groups
from data.group_sampler import HeterogeneousGroupSampler
from rewards.safety_task_reward import CompositeSafetyReward

logger = logging.getLogger(__name__)


def build_dataset(smoke_test: bool) -> tuple[HeterogeneousGroupSampler, HeterogeneousGroupSampler]:
    intents = load_advbench_intents()
    if smoke_test:
        intents = intents[:8]
        logger.info("Smoke test: truncated to %d intents", len(intents))

    id_groups, ood_groups = build_meta_groups(intents)
    logger.info("Built %d ID meta-groups, %d OOD meta-groups", len(id_groups), len(ood_groups))

    return HeterogeneousGroupSampler(id_groups), HeterogeneousGroupSampler(ood_groups)


def build_reward_function(composite_reward: CompositeSafetyReward):
    """
    Adapts CompositeSafetyReward to TRL's expected reward_funcs signature:
    a callable that takes lists of prompts/completions and returns a list
    of floats. Wrap rather than reshape the reward classes themselves, so
    they stay independently unit-testable (see tests/test_data_pipeline.py).
    """

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        is_anchor_flags = kwargs["is_anchor"]
        behaviors = kwargs["behavior"]  # now a real dataset column, see group_sampler.py
        return [
            composite_reward.score(p, c, is_anchor=a, behavior=b)
            for p, c, a, b in zip(prompts, completions, is_anchor_flags, behaviors)
        ]

    return reward_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Use DEBUG for more verbose per-step logging.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    train_dataset, ood_dataset = build_dataset(smoke_test=args.smoke_test)
    composite_reward = CompositeSafetyReward()
    reward_fn = build_reward_function(composite_reward)

    # Deferred-heavy imports: keep TRL/transformers/torch out of module import
    # time so this file stays importable and unit-testable without the full
    # training stack installed.
    import torch  # type: ignore
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        logger.warning(
            "No CUDA GPU detected, running on CPU. Fine for --smoke-test, "
            "but do NOT run the full %d-step training on CPU, move that to "
            "an HF GPU Job.",
            CONFIG.training.num_train_steps,
        )

    generation_batch_size = (
        CONFIG.training.per_device_train_batch_size * CONFIG.training.gradient_accumulation_steps
    )
    assert generation_batch_size % CONFIG.training.group_size_k == 0, (
        f"per_device_train_batch_size * gradient_accumulation_steps "
        f"({generation_batch_size}) must be divisible by group_size_k "
        f"({CONFIG.training.group_size_k}), or TRL's GRPOConfig will reject "
        f"it at construction time with a less obvious error. Adjust config.py."
    )

    grpo_config = GRPOConfig(
        output_dir=CONFIG.output_dir,
        learning_rate=CONFIG.training.learning_rate,
        num_generations=CONFIG.training.group_size_k,
        per_device_train_batch_size=CONFIG.training.per_device_train_batch_size,
        gradient_accumulation_steps=CONFIG.training.gradient_accumulation_steps,
        max_steps=2 if args.smoke_test else CONFIG.training.num_train_steps,
        seed=CONFIG.training.seed,
        bf16=has_cuda,
        use_cpu=not has_cuda,
        # FIX for CUDA OOM: bf16=True above only controls mixed-precision
        # COMPUTE, not the weights' resting dtype. Without this,
        # GRPOTrainer loads the policy model in fp32 by default (TRL's
        # model_init_kwargs defaults to None), which combined with fp32
        # AdamW optimizer states (2x model size) is what caused the OOM
        # on a10g-small (22GB) during real testing. Explicitly requesting
        # bf16 here halves both.
        model_init_kwargs={"torch_dtype": "bfloat16"} if has_cuda else None,
        # Trades compute for activation memory, cheap insurance against
        # OOM recurring at a slightly larger step/batch config later.
        gradient_checkpointing=has_cuda,
        max_completion_length=CONFIG.training.max_completion_length,
    )

    trainer = GRPOTrainer(
        model=CONFIG.model.policy_model_name,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
    )

    logger.info(
        "Starting GRPO baseline training: model=%s steps=%d smoke_test=%s",
        CONFIG.model.policy_model_name,
        grpo_config.max_steps,
        args.smoke_test,
    )
    trainer.train()
    trainer.save_model(CONFIG.output_dir)
    logger.info("Baseline training complete, saved to %s", CONFIG.output_dir)


if __name__ == "__main__":
    main()