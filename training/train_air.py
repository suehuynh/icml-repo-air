"""
Phase 2: GRPO + AIR (Anchor Invariance Regularization). This is the run
that should move OOD consistency from ~13.73% (Phase 1 GRPO baseline)
toward the paper's claimed ~60.78% (Claim 3 in the challenge README).

Run shape, same pattern as Phase 1:
    python -m training.train_air --smoke-test      # tiny local run
    python -m training.train_air                   # full config run

Before trusting a real run, re-read training/air_trainer.py's module
docstring, it documents the specific risk points (injected-column survival
through TRL's batching, the open-only auxiliary sum assumption) that need
empirical verification, not just code review.
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
        # NOTE: unlike grpo_baseline.py's smoke test, this MUST keep enough
        # intents to fill at least one full generation batch's worth of
        # complete meta-groups, or MetaGroupRepeatSampler will silently
        # produce zero batches (see its "drop trailing partial batch"
        # behavior). With per_device_train_batch_size=3, gradient_accumulation
        # _steps=3, group_size_k=3, one generation batch needs exactly 1
        # meta-group (3 intents worth isn't right, it needs 1 INTENT since
        # each intent already contributes a full 3-member group). Keep a
        # few extra intents as margin.
        intents = intents[:8]
        logger.info("Smoke test: truncated to %d intents", len(intents))

    id_groups, ood_groups = build_meta_groups(intents)
    logger.info("Built %d ID meta-groups, %d OOD meta-groups", len(id_groups), len(ood_groups))

    return HeterogeneousGroupSampler(id_groups), HeterogeneousGroupSampler(ood_groups)


def build_reward_function(composite_reward: CompositeSafetyReward):
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        is_anchor_flags = kwargs["is_anchor"]
        behaviors = kwargs["behavior"]
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
        help="Use DEBUG to see the AIR aux_loss=... grpo_loss=... line per step.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    train_dataset, ood_dataset = build_dataset(smoke_test=args.smoke_test)
    composite_reward = CompositeSafetyReward()
    reward_fn = build_reward_function(composite_reward)

    import torch  # type: ignore
    from trl import GRPOConfig  # type: ignore

    from training.air_trainer import build_air_trainer_class

    AIRTrainer = build_air_trainer_class()

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
        f"({CONFIG.training.group_size_k})."
    )
    distinct_prompts_per_generation = generation_batch_size // CONFIG.training.group_size_k

    # Derived from the REAL dataset, not from
    # anchors_per_group/open_variants_per_group config arithmetic. Those
    # config values are descriptive only and already drifted out of sync
    # with JailbreakVariantBuilder's actual template count once, this
    # avoids that exact class of bug recurring silently.
    meta_group_size = train_dataset.uniform_group_size
    assert distinct_prompts_per_generation % meta_group_size == 0, (
        f"distinct prompts per generation batch ({distinct_prompts_per_generation}) "
        f"must be divisible by the ACTUAL meta-group size ({meta_group_size}, "
        "read from the dataset itself), or MetaGroupRepeatSampler will raise. "
        "Adjust config.py's TrainingConfig batch sizes."
    )

    # CRITICAL, added after empirical testing: the two checks above are
    # necessary but NOT sufficient. They only guarantee a full generation
    # batch contains whole groups, they do NOT guarantee any single
    # micro-step (one per_device_train_batch_size-sized slice, replayed
    # gradient_accumulation_steps times) sees a COMPLETE group rather than
    # a fragment. This is the check that would have caught the "N group(s)
    # had no anchor or no open-variant member" failure before it silently
    # ran an entire training pass with AIR's auxiliary loss skipped.
    rows_per_group = meta_group_size * CONFIG.training.group_size_k
    assert CONFIG.training.per_device_train_batch_size % rows_per_group == 0, (
        f"per_device_train_batch_size ({CONFIG.training.per_device_train_batch_size}) "
        f"must itself be a whole multiple of meta_group_size * group_size_k "
        f"({meta_group_size} * {CONFIG.training.group_size_k} = {rows_per_group}), "
        "or a single micro-step's row slice can cut a group in half across "
        "the anchor/open boundary. This is what caused the AIR aux loss to "
        "silently no-op during earlier testing, adjust config.py."
    )

    grpo_config = GRPOConfig(
        output_dir=CONFIG.output_dir + "_air",
        learning_rate=CONFIG.training.learning_rate,
        num_generations=CONFIG.training.group_size_k,
        per_device_train_batch_size=CONFIG.training.per_device_train_batch_size,
        gradient_accumulation_steps=CONFIG.training.gradient_accumulation_steps,
        max_steps=2 if args.smoke_test else CONFIG.training.num_train_steps,
        seed=CONFIG.training.seed,
        bf16=has_cuda,
        use_cpu=not has_cuda,
        # Same OOM fix as grpo_baseline.py: without this, TRL loads the
        # policy model in fp32 by default (model_init_kwargs=None), and
        # fp32 weights + fp32 AdamW optimizer states is what caused the
        # CUDA OOM on a10g-small during real testing.
        model_init_kwargs={"torch_dtype": "bfloat16"} if has_cuda else None,
        gradient_checkpointing=has_cuda,
        max_completion_length=CONFIG.training.max_completion_length,
    )

    trainer = AIRTrainer(
        model=CONFIG.model.policy_model_name,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        air_lambda=CONFIG.training.air_lambda,
    )

    logger.info(
        "Starting GRPO+AIR training: model=%s steps=%d lambda=%s smoke_test=%s",
        CONFIG.model.policy_model_name,
        grpo_config.max_steps,
        CONFIG.training.air_lambda,
        args.smoke_test,
    )
    trainer.train()
    trainer.save_model(CONFIG.output_dir + "_air")
    logger.info("AIR training complete, saved to %s", CONFIG.output_dir + "_air")


if __name__ == "__main__":
    main()