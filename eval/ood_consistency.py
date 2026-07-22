"""
Out-of-distribution consistency: Acc_group computed on the OOD holdout
meta-groups. This is the paper's "invariant robustness" metric applied to
prompts/intents the model never trained on.

Comparison target: Claim 3 (challenge README) gives an exact Safety-domain
number, which is why Safety-only scope can attempt a FULL verification of
this specific claim (unlike Claim 2, which aggregates all 3 domains):

    GRPO alone:   OOD consistency 13.73%
    GRPO+AIR:     OOD consistency 60.78%

NOTE ON OOD SPLIT FIDELITY: per data/advbench_loader.py's build_meta_groups,
this repo's OOD split is a disjoint intent holdout, not the paper's
"different generation models and prompt-construction procedures". Keep
this caveat attached to any OOD number you report, it's a legitimate
simplification but not an identical procedure.
"""

from __future__ import annotations

from config import CONFIG
from data.group_sampler import HeterogeneousGroupSampler
from eval.base_evaluator import BaseEvaluator, EvaluationSummary


class OODConsistencyEvaluator(BaseEvaluator):
    def metric_name(self) -> str:
        return "OOD Consistency (Safety domain)"

    def report(self, baseline_summary: EvaluationSummary, air_summary: EvaluationSummary) -> str:
        baseline_pct = baseline_summary.as_percentages()["group_accuracy_pct"]
        air_pct = air_summary.as_percentages()["group_accuracy_pct"]
        targets = CONFIG.claim_targets

        return (
            f"{self.metric_name()}\n"
            f"{'-' * len(self.metric_name())}\n"
            f"GRPO baseline:  {baseline_pct}%  "
            f"(target: {targets.safety_ood_consistency_grpo_pct}%, "
            f"n_groups={baseline_summary.num_groups})\n"
            f"GRPO+AIR:       {air_pct}%  "
            f"(target: {targets.safety_ood_consistency_grpo_air_pct}%, "
            f"n_groups={air_summary.num_groups})\n"
            f"Measured gain:  {air_pct - baseline_pct:+.2f}pp\n"
            f"Claim 3 target gain: "
            f"+{targets.safety_ood_consistency_grpo_air_pct - targets.safety_ood_consistency_grpo_pct:.2f}pp\n"
            f"CAVEAT: OOD split here is a disjoint intent holdout, not the "
            f"paper's different-generation-model procedure, see module docstring."
        )


def run_ood_evaluation(
    model,
    tokenizer,
    ood_dataset: HeterogeneousGroupSampler,
    judge=None,
    max_new_tokens: int = 128,
) -> EvaluationSummary:
    evaluator = OODConsistencyEvaluator(judge=judge)
    return evaluator.evaluate(model, tokenizer, ood_dataset, max_new_tokens=max_new_tokens)