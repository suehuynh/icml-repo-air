# Reproducing AIR: Towards Context-Invariant Safety Alignment for Large Language Models

Reproduction attempt for the ICML 2026 Agent Reproduction Challenge, targeting [ICML 2026 paper #5311](https://openreview.net/forum?id=7Eyp6J31u4), "Towards Context-Invariant Safety Alignment for Large Language Models" (OpenReview ID: `7Eyp6J31u4`).

Challenge: [ICML-2026-agent-repro](https://huggingface.co/ICML-2026-agent-repro)

## What this paper claims

The paper introduces **Anchor Invariance Regularization (AIR)**, a technique for making RLHF-style alignment robust across different presentations of the same underlying request. Verifiable prompts (multiple-choice, rule-checkable formats) are treated as anchors with a stop-gradient target; open-ended variants of the same latent intent are regularized toward the anchor's behavior via an auxiliary loss on top of GRPO, rather than the reverse. The stated goal is to reduce reward hacking and superficial compliance that appears when a model behaves safely in an easily-verified context but not in a paraphrased or adversarially-framed one.

## Scope of this reproduction

The paper evaluates across three domains (Safety, Moral Reasoning, Mathematics). **This reproduction is scoped to the Safety domain only**, using AdvBench, to keep the attempt fully verifiable within a small compute budget. This means:

- Claims specific to Safety (see Claim 3 below) can be **fully** attempted.
- Claims that aggregate across all three domains (see Claim 2 below) receive **partial** evidence only, not full verification. This is stated explicitly in the eval reports, not implied as complete coverage.

### Claims under test (from the official challenge README)

1. AIR's stop-gradient mechanism (Section 3.3, Equations 5-6): open-ended variants regularized toward anchors, not the reverse.
2. Aggregate gains across all three domains: +12.71pp in-distribution group accuracy, +33.49pp OOD consistency (Table 1). **Safety-only scope means this repo contributes partial evidence toward this claim.**
3. Safety domain specifically: OOD consistency rises from 13.73% (GRPO) to 60.78% (GRPO+AIR) (Table 1). **This is the primary target for this reproduction.**
4. Domain/dataset construction (Section 4): confirmed for Safety (AdvBench + jailbreak variants vs. rule-checkable anchor format).
5. Robustness to reward hacking under a corrupted reward signal (Section 4.2, Figure 5): not yet attempted, a candidate extension if time and budget allow.

## Documented substitutions

Per the challenge FAQ, a documented backbone/scale substitution counts as a full reproduction as long as the substituted component isn't the paper's actual contribution. The following are deliberate, budget-driven substitutions from this paper's Section 4.1 and Appendix C:

| Component | Paper | This reproduction |
|---|---|---|
| Policy model | Qwen-2.5-14B | Qwen2.5-1.5B-Instruct |
| Judge model | Frozen copy of the policy model | Same (frozen copy of the 1.5B policy) |
| Final eval judge | GPT-4.1 | Open-source substitute (see `config.py`, `final_eval_judge_is_external_api=False`) |
| Training steps | 3,000 | Configurable in `config.py`, scaled down for budget, see the logbook for the actual value used in the reported run |

Hyperparameters preserved exactly from the paper: `K=3` (group size), learning rate `5e-7`, AIR coefficient `λ=8e-4`.

## Repository structure

```
config.py                    # single source of truth: hyperparameters, reward tiers, claim targets
data/
  advbench_loader.py          # anchor + jailbreak-variant construction on top of AdvBench
  group_sampler.py             # flattens meta-groups into a GRPOTrainer-compatible dataset
rewards/
  format_reward.py             # CoT + answer-tag structural reward (Appendix C.1)
  safety_task_reward.py        # 10-facet judge + friendliness tier, anchor rule-grading (Appendix C.2.1)
training/
  grpo_baseline.py             # Phase 1: standard GRPO, no AIR
  samplers.py                   # MetaGroupRepeatSampler: keeps anchor + open variants together per batch
  air_trainer.py                # Phase 2: AIRTrainer, adds the J_aux auxiliary loss
  train_air.py                  # Phase 2 runner
eval/
  base_evaluator.py             # shared Acc / Acc_group computation
  anchor_accuracy.py            # ID-split evaluator (Claim 2, partial evidence)
  ood_consistency.py            # OOD-split evaluator (Claim 3, primary target)
  run_evaluation.py             # loads both checkpoints, prints logbook-ready comparison reports
scripts/
  verify_advbench_mirror.py     # confirms the AdvBench HF mirror matches the canonical 520-behavior set
tests/                         # structural unit tests, no GPU required
```

## Setup

```bash
pip install -r requirements.txt
python -m scripts.verify_advbench_mirror   # confirm the data source
pytest tests/ -v                            # structural tests, no GPU needed
```

## Running

```bash
# Phase 1: GRPO baseline
python -m training.grpo_baseline --smoke-test      # local sanity check
python -m training.grpo_baseline                    # full run (GPU recommended)

# Phase 2: GRPO + AIR
python -m training.train_air --smoke-test --log-level DEBUG
python -m training.train_air

# Evaluation, once both checkpoints exist
python -m eval.run_evaluation
```

Full-scale training was run on a Hugging Face GPU Job. See the Trackio logbook (linked from the challenge submission) for the job URL, hardware, configuration, and results.

## Known open assumptions

A few implementation details were not fully specified in the paper excerpts available during this reproduction and were filled in with a defensible, documented choice rather than left unimplemented. These are called out inline in the relevant source files (`data/advbench_loader.py`, `rewards/safety_task_reward.py`, `training/air_trainer.py`) and summarized here:

- Exact jailbreak template wording (paper doesn't publish its verbatim set for AdvBench).
- Facet aggregation rule across the 10 safety facets (this implementation uses worst-case OR: any facet flagging Unsafe makes the overall verdict Unsafe).
- Whether AIR's auxiliary loss sums over open-variant completions only or all completions (this implementation sums over open variants only; Equations 5-6 in Section 3.3 would confirm this directly if accessible).
- OOD split construction: this repo holds out a disjoint subset of intents, rather than the paper's approach of using different generation models and prompt-construction procedures for the ID/OOD split.

## Author

Nguyen (Sue) Huynh, MSc Data Science student at Brown University.