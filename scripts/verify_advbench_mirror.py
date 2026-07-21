"""
Verifies the configured AdvBench HF mirror matches the canonical dataset
before you trust anything built on top of it. Run this ONCE before Phase 1
training, log the output in your Trackio logbook.

    python -m scripts.verify_advbench_mirror

Checks:
  1. Row count matches the canonical 520 harmful behaviors
     (llm-attacks/llm-attacks, data/advbench/harmful_behaviors.csv).
  2. The first row's `goal` field matches the canonical first entry, which
     catches mirrors that silently reformat, filter, or paraphrase entries.

This does NOT check all 520 rows against the source CSV (that requires
fetching the GitHub file, which may not be in your network allowlist
depending on environment). If you can fetch it, extend this script to do a
full row-by-row diff instead of just the row-count + first-row spot check.
"""

from __future__ import annotations

from config import CONFIG


def verify() -> None:
    from datasets import load_dataset  # deferred-heavy import

    ds = load_dataset(CONFIG.data.advbench_hf_repo, split="train")

    actual_count = len(ds)
    expected_count = CONFIG.data.expected_advbench_row_count
    count_ok = actual_count == expected_count

    first_row = ds[0]
    actual_first_goal = first_row.get("goal") or first_row.get("prompt") or first_row.get("behavior")
    expected_first_goal = CONFIG.data.expected_first_goal
    first_row_ok = actual_first_goal == expected_first_goal

    print(f"Repo:                 {CONFIG.data.advbench_hf_repo}")
    print(f"Row count:            {actual_count} (expected {expected_count}) -> {'OK' if count_ok else 'MISMATCH'}")
    print(f"First goal matches:   {first_row_ok}")
    if not first_row_ok:
        print(f"  Expected: {expected_first_goal!r}")
        print(f"  Actual:   {actual_first_goal!r}")
    print(f"Columns present:      {list(first_row.keys())}")

    if count_ok and first_row_ok:
        print("\nVERDICT: mirror appears to match the canonical AdvBench set.")
    else:
        print(
            "\nVERDICT: mirror does NOT clearly match canonical AdvBench. "
            "Investigate before training, and document whatever you find "
            "(and whichever mirror you end up using) in the logbook."
        )


if __name__ == "__main__":
    verify()