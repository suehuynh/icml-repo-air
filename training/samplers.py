"""
TRL's default `RepeatSampler` (trl/trainer/utils.py) shuffles across
individual dataset rows with no concept of intent_id grouping. For AIR,
that's a real bug, not a style issue: if an anchor and its open variants
don't land in the same batch, there's nothing to compute r_bar_acr / r_bar_c
against for that step, and the auxiliary loss silently degrades toward a
no-op for however many steps that happens.

This sampler mirrors RepeatSampler's exact contract (same constructor
signature semantics, same flat-index-sequence output), but samples whole
meta-groups as its atomic unit instead of individual prompts:

  RepeatSampler:          picks `batch_size` distinct prompt indices, each
                          repeated `mini_repeat_count` times.
  MetaGroupRepeatSampler: picks `batch_size // group_size` distinct
                          intent_ids, yields ALL of each one's member
                          indices (contiguously), each repeated
                          `mini_repeat_count` times.

This guarantees that whatever downstream chunking TRL applies to the flat
index sequence (per_device_train_batch_size-sized slices, where that size
is in ROW units already including the mini_repeat_count/K repeats, NOT
distinct-prompt units, confirmed empirically after an earlier version of
this docstring got that wrong), each chunk contains one or more COMPLETE
meta-groups, never a fragment, PROVIDED per_device_train_batch_size is
itself a whole multiple of (group_size * mini_repeat_count). That
additional check lives in training/train_air.py, not here, since this
class only controls the flat ordering, not how TRL later slices it.
"""

from __future__ import annotations

from typing import Sized

try:
    import torch
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover
    # Structural tests (grouping/ordering logic) don't need torch, only the
    # real training run does. Fall back so this module stays importable and
    # unit-testable without the full ML stack installed.
    torch = None  # type: ignore
    Sampler = object  # type: ignore


class MetaGroupRepeatSampler(Sampler):
    """
    Args mirror RepeatSampler exactly, see its docstring in
    trl/trainer/utils.py. `group_ids` and `group_size` are the only
    additions: `group_ids` is the dataset's list of distinct intent_ids
    (HeterogeneousGroupSampler.group_ids), `group_size` is the uniform
    number of examples per group (HeterogeneousGroupSampler.uniform_group_size).

    CRITICAL PRECONDITION, not just a nice-to-have: `batch_size` (the
    distinct-prompt count TRL derives internally as
    generation_batch_size // num_generations) MUST be a whole multiple of
    group_size. This is necessary but NOT sufficient on its own, see
    training/train_air.py's additional assertion on
    per_device_train_batch_size itself, which is the check that actually
    prevents a micro-step's row slice from cutting a group in half. If
    either isn't satisfied, __init__ raises rather than silently producing
    batches with split meta-groups, which would make the AIR loss wrong in
    a way that's easy to miss (it would run without error, just compute a
    meaningless gap for whichever groups got split, or silently skip the
    auxiliary loss entirely for that step).
    """

    def __init__(
        self,
        group_ids: list[str],
        group_size: int,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        if batch_size % group_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a whole multiple of "
                f"group_size ({group_size}), or meta-groups will be split "
                "across generation batches. Adjust per_device_train_batch_size "
                "/ gradient_accumulation_steps / num_generations in config.py "
                "so (per_device_train_batch_size * gradient_accumulation_steps) "
                "// num_generations is a multiple of group_size."
            )

        self.group_ids = group_ids
        self.group_size = group_size
        self.num_groups_per_batch = batch_size // group_size
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_groups = len(group_ids)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

    def _group_indices_in_order(self, group_order: list[int], group_lookup) -> list[int]:
        """Flattens a list of group positions into their member dataset indices, in order."""
        flat: list[int] = []
        for group_pos in group_order:
            intent_id = self.group_ids[group_pos]
            flat.extend(group_lookup(intent_id))
        return flat

    def __iter__(self):
        # Deferred import: avoids a hard torch dependency for anything that
        # only inspects this sampler's construction logic in tests.
        if self.shuffle:
            group_order = torch.randperm(self.num_groups, generator=self.generator).tolist()
        else:
            group_order = list(range(self.num_groups))

        # Chunk the shuffled group order into batches of num_groups_per_batch,
        # dropping a trailing partial batch (matches RepeatSampler's own
        # truncation behavior for a partial final batch).
        for start in range(0, self.num_groups - self.num_groups_per_batch + 1, self.num_groups_per_batch):
            batch_group_positions = group_order[start:start + self.num_groups_per_batch]
            for _ in range(self.repeat_count):
                for group_pos in batch_group_positions:
                    intent_id = self.group_ids[group_pos]
                    yield from (idx for idx in self._member_indices(intent_id) for _ in range(self.mini_repeat_count))

    def _member_indices(self, intent_id: str):
        raise NotImplementedError(
            "Set via bind_dataset() before iterating, see training/air_trainer.py."
        )

    def bind_dataset(self, dataset) -> "MetaGroupRepeatSampler":
        """
        Wires this sampler to a HeterogeneousGroupSampler's group->indices
        lookup. Separate from __init__ so this class can be constructed and
        unit-tested with a plain list of fake group_ids, without needing a
        real dataset instance.
        """
        self._member_indices = dataset.indices_for_group  # type: ignore[method-assign]
        return self

    def __len__(self) -> int:
        num_full_batches = self.num_groups // self.num_groups_per_batch
        return num_full_batches * self.batch_size * self.mini_repeat_count * self.repeat_count