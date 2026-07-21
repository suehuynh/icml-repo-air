"""
In-distribution accuracy: Acc and Acc_group computed on the ID meta-groups
(the non-holdout split from data.advbench_loader.build_meta_groups).

Comparison target: Claim 2 (challenge README) reports a +12.71pp aggregate
ID group accuracy gain across all 3 domains (Safety + Moral Reasoning +
Math). Since this repro is Safety-only, a measured gain here is PARTIAL
evidence toward Claim 2, not a full verification of it, say so explicitly
in your logbook rather than implying full coverage.
"""

from __future__ import annotations

from config import CONFIG
from data.group_sampler import HeterogeneousGroupSampler
from eval.base_evaluator import BaseEvaluator, EvaluationSummary


class AnchorAccuracyEvaluator(BaseEvaluator):
    def metric_name(self) -> str:
        return "ID Group Accuracy (Safety domain)"

    def report(self, baseline_summary: EvaluationSummary, air_summary: EvaluationSummary) -> str:
        """
        Formats a side-by-side comparison against Claim 2's target, for
        pasting directly into a Trackio logbook page.
        """
        baseline_pct = baseline_summary.as_percentages()
        air_pct = air_summary.as_percentages()
        gain_pp = air_pct["group_accuracy_pct"] - baseline_pct["group_accuracy_pct"]

        return (
            f"{self.metric_name()}\n"
            f"{'-' * len(self.metric_name())}\n"
            f"GRPO baseline:  Acc={baseline_pct['prompt_accuracy_pct']}%  "
            f"Acc_group={baseline_pct['group_accuracy_pct']}%  "
            f"(n_groups={baseline_summary.num_groups}, n_prompts={baseline_summary.num_prompts})\n"
            f"GRPO+AIR:       Acc={air_pct['prompt_accuracy_pct']}%  "
            f"Acc_group={air_pct['group_accuracy_pct']}%  "
            f"(n_groups={air_summary.num_groups}, n_prompts={air_summary.num_prompts})\n"
            f"Measured gain:  {gain_pp:+.2f}pp (group accuracy)\n"
            f"Claim 2 target: +{CONFIG.claim_targets.id_group_accuracy_gain_pp}pp "
            f"(aggregate across all 3 domains, this is Safety-only, partial evidence)"
        )


def run_id_evaluation(
    model,
    tokenizer,
    id_dataset: HeterogeneousGroupSampler,
    judge=None,
) -> EvaluationSummary:
    """Convenience entry point: instantiate evaluator, run, return summary."""
    evaluator = AnchorAccuracyEvaluator(judge=judge)
    return evaluator.evaluate(model, tokenizer, id_dataset)