"""
Heterogeneous prompt grouping (paper §3.5), turning MetaGroups into the
flat, GRPO-Trainer-compatible dataset format while preserving the
group -> {anchor, open} membership needed to compute the AIR auxiliary loss.

Paper: "our data loader does not sample independent prompts. Instead, for
each latent task instance z ... we construct a meta-group S_z = {s_anchor^1,
..., s_anchor^m, s_open^1, ..., s_open^n}. During each training step, we
sample a subset of prompts from S_z encompassing both types."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from data.advbench_loader import MetaGroup

try:
    # Real training needs torch.utils.data.Dataset for TRL/DataLoader
    # compatibility. Structural unit tests (grouping logic, no model calls)
    # should not require torch to be installed, so fall back to plain
    # `object` if it's unavailable and only fail loudly if training code
    # actually tries to use Dataset-specific behavior.
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object  # type: ignore


@dataclass(frozen=True)
class GroupedPrompt:
    """
    One training example, tagged with enough metadata to reconstruct which
    meta-group and which context (anchor vs open) it belongs to. This is
    what the AIR loss needs at train time: r_bar_acr vs r_bar_c per group.
    """

    intent_id: str
    prompt: str
    is_anchor: bool
    behavior: str  # raw AdvBench goal, needed by the judge, not the rendered prompt


class HeterogeneousGroupSampler(Dataset):
    """
    Flattens MetaGroups into individual (prompt, is_anchor, intent_id)
    examples, while keeping a group_id -> indices index so the trainer's
    reward/loss code can regroup them per training step.

    Usage with TRL's GRPOTrainer: pass this as `train_dataset`. TRL will
    tokenize the `prompt` column; keep `intent_id` and `is_anchor` as extra
    columns for the reward function and the AIR auxiliary loss to consume
    (see rewards/safety_task_reward.py and training/grpo_baseline.py).
    """

    def __init__(self, meta_groups: list[MetaGroup]) -> None:
        self._examples: list[GroupedPrompt] = []
        self._group_index: dict[str, list[int]] = {}
        self._intent_id_to_idx: dict[str, int] = {}

        for group in meta_groups:
            self._intent_id_to_idx.setdefault(group.intent_id, len(self._intent_id_to_idx))
            start = len(self._examples)
            for anchor_prompt in group.anchor_prompts:
                self._examples.append(
                    GroupedPrompt(
                        intent_id=group.intent_id,
                        prompt=anchor_prompt,
                        is_anchor=True,
                        behavior=group.behavior,
                    )
                )
            for open_prompt in group.open_prompts:
                self._examples.append(
                    GroupedPrompt(
                        intent_id=group.intent_id,
                        prompt=open_prompt,
                        is_anchor=False,
                        behavior=group.behavior,
                    )
                )
            self._group_index[group.intent_id] = list(range(start, len(self._examples)))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        return {
            "prompt": ex.prompt,
            "intent_id": ex.intent_id,
            "is_anchor": ex.is_anchor,
            "behavior": ex.behavior,
            # Integer form of intent_id, needed because GRPOTrainer's internal
            # batching/slicing machinery operates on tensors. A plain string
            # column risks not surviving that machinery the same way tensor
            # columns do, see training/air_trainer.py for where this is used.
            "intent_idx": self._intent_id_to_idx[ex.intent_id],
        }

    def indices_for_group(self, intent_id: str) -> list[int]:
        return self._group_index[intent_id]

    def iter_groups(self) -> Iterator[tuple[str, list[GroupedPrompt]]]:
        for intent_id, indices in self._group_index.items():
            yield intent_id, [self._examples[i] for i in indices]

    @property
    def group_ids(self) -> list[str]:
        """All intent_ids, in first-seen order. Used by the meta-group-aware sampler."""
        return list(self._group_index.keys())

    @property
    def uniform_group_size(self) -> int:
        """
        Returns the common meta-group size (anchors + opens) if every group is
        the same size, raises otherwise. AIRTrainer's sampler currently
        assumes uniform group size, this is the guard that catches it if
        that assumption is ever violated (e.g. a future change to
        anchors_per_group/open_variants_per_group that isn't applied
        consistently).
        """
        sizes = {len(indices) for indices in self._group_index.values()}
        if len(sizes) != 1:
            raise ValueError(
                f"Meta-groups have inconsistent sizes: {sizes}. "
                "MetaGroupRepeatSampler assumes every group is the same "
                "size, update it before training if this is intentional."
            )
        return sizes.pop()