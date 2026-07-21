# Run the reproduction: pick a harness
1. Install the OpenResearch CLI
`curl -LsSf https://openresearch.sh/install.sh | sh && source "$HOME/.cargo/env"`
2. Launch the dashboard and create a new Blank Project
`orx up`
3. Add a HF write token to the OpenResearch settings: Environment -> HF_TOKEN (paste value)`

# Paste this into a new OpenResearch session. It will read the full challenge guide and publish the Trackio logbook.
`# First, read the challenge instructions:
curl -sL https://huggingface.co/datasets/ICML-2026-agent-repro/challenge/resolve/main/README.md

Your job is to reproduce the ICML 2026 paper #5311 — Towards Context-Invariant Safety Alignment for Large Language Models (OpenReview id: 7Eyp6J31u4). Here are the major claims that you should verify:
- Claim 1: Anchor Invariance Regularization (AIR) designates verifiable contexts (e.g., multiple-choice formulations) as anchors with stop-gradient targets, so open-ended variants are regularized toward the anchor rather than the reverse (Section 3.3, Equations 5-6).
- Claim 2: GRPO+AIR improves in-distribution group accuracy by 12.71 percentage points and out-of-distribution consistency by 33.49 percentage points relative to GRPO alone (Table 1, Section 4.2).
- Claim 3: On the Safety domain specifically, out-of-distribution consistency rises from 13.73% (GRPO) to 60.78% (GRPO+AIR) (Table 1).
- Claim 4: Evaluation spans three domains: Safety (AdvBench with jailbreak variants vs. rule-checkable formats), Moral Reasoning (ethical dilemmas with verifiable anchors), and Mathematics (GSM8K converted to multiple-choice anchors) (Section 4).
- Claim 5: Under a deliberately corrupted reward signal (a naive regex heuristic rewarding superficial refusal keywords), GRPO+AIR resists alignment faking while plain GRPO exhibits severe reward hacking (Section 4.2, "Robustness to Extreme Reward Hacking", Figure 5).

Try to verify the claims as much as you can locally or using Hugging Face Jobs. After you are done, publish the Trackio logbook and print the link here.`