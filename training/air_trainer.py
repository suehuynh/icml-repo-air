"""
AIRTrainer implements the paper's auxiliary loss on top of TRL's GRPOTrainer:

    J_aux = -(1/N) * sum_i [ (r_bar_acr - r_bar_c) * r_i * log(pi_theta(y_i|s_i)) ]
    L_total = L_policy + lambda * J_aux

where r_bar_acr is the anchor's mean reward within a meta-group
(stop-gradient), r_bar_c is the open-variant mean reward within the same
group, and the sum is over open-variant completions (see ASSUMPTION 1
below, this specific detail isn't confirmed from the paper excerpts
available, verify against Equations 5-6 in Section 3.3 if you can get
access to them, Claim 1 in the challenge README points there directly).

============================================================================
THE HIGHEST-RISK PART OF THIS FILE: injected batch metadata may not survive
TRL's internal buffering.
============================================================================
GRPOTrainer generates a full "generation batch" at once, then replays
slices of it across `steps_per_generation` training steps (see
_get_train_sampler's docstring/ASCII diagram in trl/trainer/grpo_trainer.py).
That replay buffer is TRL-internal (`self._buffered_inputs`), and this file
injects extra tensor columns (air_raw_reward, air_is_anchor, air_intent_idx)
into the dict returned by `_generate_and_score_completions`, HOPING TRL's
buffering slices all dict entries uniformly by batch dimension, the same
way it slices `advantages`/`prompt_ids`/etc.

This is a reasonable bet (the buffering code that's been inspected sections
of does look like generic dict-of-tensors slicing) but has NOT been
empirically verified end-to-end against a real multi-step training run.
`_compute_loss` below asserts the injected columns are present and
shape-aligned with `advantages` on every single call specifically so this
fails LOUDLY the first time it's wrong, instead of silently training on
misaligned reward/group data. DO NOT remove those assertions to "fix" a
crash, if they fire, the fix is in the buffering/injection mechanism, not
in silencing the check.

============================================================================
ASSUMPTIONS (verify before trusting results):
============================================================================
1. J_aux sums over OPEN-VARIANT completions only, not anchors. Anchors
   already get an ordinary policy gradient from the main GRPO loss; AIR's
   stated purpose ("open-ended variants are regularized toward the anchor
   rather than the reverse") reads as an asymmetric correction applied to
   the open side. Not confirmed against Equations 5-6 directly.
2. r_bar_c (open-variant mean reward) is not itself given an explicit
   stop-gradient in the paper excerpt. In this implementation it's moot:
   all reward values come from an external judge and were already
   `.detach()`-ed at capture time in `_calculate_rewards`, so no gradient
   flows through ANY reward term regardless, only through
   log(pi_theta(y_i|s_i)) as in a standard policy-gradient loss.
3. This implementation assumes single-device training (no distributed /
   multi-GPU). Multi-GPU introduces a gather-then-process-slice step
   between reward computation and loss computation that would need
   separate handling, not attempted here since your GPU Job is a single
   device.
"""

from __future__ import annotations

import logging

from config import CONFIG

logger = logging.getLogger(__name__)


def _lazy_imports():
    """Deferred-heavy: keeps this module importable without torch/trl installed."""
    import torch
    from trl import GRPOTrainer

    return torch, GRPOTrainer


def build_air_trainer_class():
    """
    Returns the AIRTrainer class, built lazily so importing this module
    doesn't require torch/trl to be installed (matches the deferred-import
    pattern used throughout this repo, see rewards/safety_task_reward.py).
    """
    torch, GRPOTrainer = _lazy_imports()
    from training.samplers import MetaGroupRepeatSampler

    class AIRTrainer(GRPOTrainer):
        def __init__(self, *args, air_lambda: float = CONFIG.training.air_lambda, **kwargs):
            super().__init__(*args, **kwargs)
            self.air_lambda = air_lambda
            self._air_last_raw_rewards = None
            self._air_last_is_anchor = None
            self._air_last_intent_idx = None

            if not (hasattr(self.train_dataset, "group_ids") and hasattr(self.train_dataset, "indices_for_group")):
                raise TypeError(
                    "AIRTrainer requires train_dataset to expose .group_ids / "
                    ".indices_for_group / .uniform_group_size, i.e. a "
                    f"HeterogeneousGroupSampler. Got {type(self.train_dataset)}."
                )

        # --- Sampler override: keeps meta-groups intact within a batch ----

        def _get_train_sampler(self, dataset=None):
            if dataset is None:
                dataset = self.train_dataset

            group_size = dataset.uniform_group_size
            # Mirror TRL's own formula exactly (trl/trainer/grpo_trainer.py,
            # _get_train_sampler) rather than re-deriving batch-size units
            # ourselves, so this stays correct regardless of exactly how
            # generation_batch_size/per_device_train_batch_size interact
            # internally, which differs across TRL versions.
            batch_size = self.args.generation_batch_size // self.num_generations

            sampler = MetaGroupRepeatSampler(
                group_ids=dataset.group_ids,
                group_size=group_size,
                mini_repeat_count=self.num_generations,
                batch_size=batch_size,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )
            return sampler.bind_dataset(dataset)

        # --- Reward capture: stash raw (pre-advantage) reward + group meta -

        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

            reward_weights = self.reward_weights.to(rewards_per_func.device)
            raw_rewards = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)

            self._air_last_raw_rewards = raw_rewards.detach()
            self._air_last_is_anchor = torch.tensor(
                [bool(ex["is_anchor"]) for ex in inputs], device=raw_rewards.device
            )
            self._air_last_intent_idx = torch.tensor(
                [int(ex["intent_idx"]) for ex in inputs], device=raw_rewards.device, dtype=torch.long
            )
            return rewards_per_func

        # --- Inject captured metadata into the batch dict TRL passes along -

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            output["air_raw_reward"] = self._air_last_raw_rewards
            output["air_is_anchor"] = self._air_last_is_anchor
            output["air_intent_idx"] = self._air_last_intent_idx
            return output

        # --- Loss: standard GRPO term + AIR auxiliary term -----------------

        def _compute_loss(self, model, inputs):
            grpo_loss = super()._compute_loss(model, inputs)

            for key in ("air_raw_reward", "air_is_anchor", "air_intent_idx"):
                if key not in inputs or inputs[key] is None:
                    raise RuntimeError(
                        f"AIRTrainer._compute_loss: '{key}' missing from inputs. "
                        "TRL's internal batching likely dropped an injected "
                        "column, see this file's module docstring, this is "
                        "the flagged highest-risk integration point. Do not "
                        "silently continue without this, results would be "
                        "meaningless."
                    )

            batch_size = inputs["air_raw_reward"].shape[0]
            if inputs["advantages"].shape[0] != batch_size:
                raise RuntimeError(
                    f"AIRTrainer._compute_loss: air_raw_reward has {batch_size} "
                    f"rows but advantages has {inputs['advantages'].shape[0]}. "
                    "Injected AIR columns are misaligned with the rest of the "
                    "batch. Do not trust any results until this is fixed."
                )

            aux_loss = self._compute_air_auxiliary_loss(model, inputs)
            logger.debug("AIR aux_loss=%.6f grpo_loss=%.6f", aux_loss.item(), grpo_loss.item())
            return grpo_loss + self.air_lambda * aux_loss

        def _compute_air_auxiliary_loss(self, model, inputs):
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            # Second forward pass through `model`, separate from the one
            # inside super()._compute_loss(). Wasteful (~2x compute for the
            # loss step only, not the far more expensive generation step),
            # but far simpler and safer than threading per_token_logps out
            # of the parent method and risking a silent mismatch if TRL's
            # internals change. Acceptable for a small-scale reproduction.
            per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
            )
            seq_logp = (per_token_logps * completion_mask).sum(dim=1)  # (B,), grad-enabled

            raw_reward = inputs["air_raw_reward"]   # (B,), already detached
            is_anchor = inputs["air_is_anchor"]      # (B,) bool
            intent_idx = inputs["air_intent_idx"]    # (B,) long

            total_loss = torch.zeros((), device=seq_logp.device, dtype=seq_logp.dtype)
            num_open_examples = 0
            num_groups_seen = 0
            num_groups_skipped = 0

            for group_id in torch.unique(intent_idx):
                group_mask = intent_idx == group_id
                anchor_mask = group_mask & is_anchor
                open_mask = group_mask & (~is_anchor)

                if anchor_mask.sum() == 0 or open_mask.sum() == 0:
                    # Should not happen if MetaGroupRepeatSampler is working
                    # correctly, since it guarantees whole groups per batch.
                    # Not raising here (a single degenerate group shouldn't
                    # kill a whole training step), but this is worth
                    # investigating if num_groups_skipped is ever > 0.
                    num_groups_skipped += 1
                    continue

                num_groups_seen += 1
                r_bar_acr = raw_reward[anchor_mask].mean()  # stop-gradient per paper (moot, see ASSUMPTION 2)
                r_bar_c = raw_reward[open_mask].mean()
                gap = r_bar_acr - r_bar_c

                # ASSUMPTION 1 (see module docstring): sum over open-variant
                # completions only.
                open_indices = open_mask.nonzero(as_tuple=True)[0]
                for idx in open_indices:
                    total_loss = total_loss - gap * raw_reward[idx] * seq_logp[idx]
                    num_open_examples += 1

            if num_groups_skipped > 0:
                logger.warning(
                    "AIR aux loss: %d group(s) in this batch had no anchor or "
                    "no open-variant member present, skipped. This should not "
                    "happen if MetaGroupRepeatSampler is wired correctly, "
                    "investigate before trusting results.",
                    num_groups_skipped,
                )

            if num_open_examples == 0:
                return torch.zeros((), device=seq_logp.device, dtype=seq_logp.dtype)

            return total_loss / num_open_examples

    return AIRTrainer