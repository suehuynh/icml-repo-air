"""
Structural tests for MetaGroupRepeatSampler: validates that meta-groups are
never split across batches and that the flat index sequence has the
expected shape. Uses shuffle=False so these run without torch installed
(torch.Generator/randperm is only exercised in the shuffle=True path, which
needs a real training environment to test meaningfully anyway).
"""

from __future__ import annotations

import pytest

from training.samplers import MetaGroupRepeatSampler


class FakeGroupedDataset:
    """Minimal stand-in for HeterogeneousGroupSampler's group indexing API."""

    def __init__(self, group_to_indices: dict[str, list[int]]) -> None:
        self._group_to_indices = group_to_indices

    def indices_for_group(self, intent_id: str) -> list[int]:
        return self._group_to_indices[intent_id]


# 6 groups (g0..g5), each with 3 members (matches anchors_per_group=1 +
# open_variants_per_group=2), indices assigned as if flattened in order.
GROUP_IDS = [f"g{i}" for i in range(6)]
GROUP_TO_INDICES = {f"g{i}": [i * 3, i * 3 + 1, i * 3 + 2] for i in range(6)}
FAKE_DATASET = FakeGroupedDataset(GROUP_TO_INDICES)


class TestConstructorValidation:
    def test_batch_size_not_multiple_of_group_size_raises(self) -> None:
        with pytest.raises(ValueError, match="whole multiple of"):
            MetaGroupRepeatSampler(
                group_ids=GROUP_IDS, group_size=3, mini_repeat_count=3, batch_size=4, shuffle=False
            )

    def test_valid_batch_size_does_not_raise(self) -> None:
        MetaGroupRepeatSampler(
            group_ids=GROUP_IDS, group_size=3, mini_repeat_count=3, batch_size=6, shuffle=False
        )


class TestIterationOrder:
    def test_never_splits_a_group_across_the_batch(self) -> None:
        """
        With batch_size=6 (2 groups per batch) and mini_repeat_count=3, each
        yielded chunk of 9 indices (3 members * 3 repeats) must belong to
        exactly one group, never straddle two.
        """
        sampler = MetaGroupRepeatSampler(
            group_ids=GROUP_IDS,
            group_size=3,
            mini_repeat_count=3,
            batch_size=6,
            repeat_count=1,
            shuffle=False,
        ).bind_dataset(FAKE_DATASET)

        flat = list(sampler)
        chunk_size = 3 * 3  # group_size * mini_repeat_count
        for start in range(0, len(flat), chunk_size):
            chunk = flat[start:start + chunk_size]
            group_of_index = {idx: idx // 3 for idx in range(18)}  # 6 groups * 3 members
            groups_in_chunk = {group_of_index[idx] for idx in chunk}
            assert len(groups_in_chunk) == 1, f"Chunk {chunk} spans multiple groups: {groups_in_chunk}"

    def test_each_member_repeated_mini_repeat_count_times_contiguously(self) -> None:
        sampler = MetaGroupRepeatSampler(
            group_ids=GROUP_IDS, group_size=3, mini_repeat_count=3, batch_size=3, shuffle=False
        ).bind_dataset(FAKE_DATASET)

        flat = list(sampler)
        # First group (g0, indices 0,1,2), each repeated 3 times contiguously.
        assert flat[:9] == [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def test_repeat_count_replays_the_same_batch(self) -> None:
        sampler = MetaGroupRepeatSampler(
            group_ids=GROUP_IDS,
            group_size=3,
            mini_repeat_count=1,
            batch_size=3,
            repeat_count=2,
            shuffle=False,
        ).bind_dataset(FAKE_DATASET)

        flat = list(sampler)
        # One batch's worth (3 indices) should appear twice before advancing
        # to the next batch of groups.
        assert flat[:3] == flat[3:6]

    def test_drops_trailing_partial_batch(self) -> None:
        # 6 groups, batch_size=6 (2 groups/batch) -> exactly 3 full batches,
        # no remainder to worry about here; use group count not evenly
        # divisible to check truncation instead.
        odd_group_ids = GROUP_IDS[:5]  # 5 groups, batch wants 2 at a time -> 2 full batches, 1 dropped
        sampler = MetaGroupRepeatSampler(
            group_ids=odd_group_ids, group_size=3, mini_repeat_count=1, batch_size=6, shuffle=False
        ).bind_dataset(FAKE_DATASET)

        flat = list(sampler)
        assert len(flat) == 2 * 6  # 2 full batches of 6, the 5th group is dropped


class TestLength:
    def test_len_matches_actual_iteration_count(self) -> None:
        sampler = MetaGroupRepeatSampler(
            group_ids=GROUP_IDS,
            group_size=3,
            mini_repeat_count=2,
            batch_size=6,
            repeat_count=3,
            shuffle=False,
        ).bind_dataset(FAKE_DATASET)

        assert len(sampler) == len(list(sampler))