"""
Cheap diagnostic, NOT part of the main eval pipeline. Loads ONE checkpoint,
generates completions for a small handful of prompts, and prints the RAW
text with no extraction or judging. Purpose: visually confirm whether the
<think>...</think><answer>...</answer> format is present at all in real
output, before spending more budget guessing why eval scored everything
as incorrect.

    python -m eval.inspect_completions --checkpoint suehuynh/icml-air-repro-baseline --num-prompts 3

Much cheaper than a full eval run: no judge model loaded, no facet
scoring, just a few generations.
"""

from __future__ import annotations

import argparse
import logging

from data.advbench_loader import load_advbench_intents, build_meta_groups

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Local path or Hub repo ID.")
    parser.add_argument("--num-prompts", type=int, default=3, help="How many prompts to sample (kept small on purpose).")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    has_cuda = torch.cuda.is_available()
    logger.info("Loading %s ...", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
        token=True,
    )
    model.eval()

    intents = load_advbench_intents()
    id_groups, _ = build_meta_groups(intents, ood_holdout_fraction=0.0)
    sample_group = id_groups[0]
    sample_prompts = [
        ("ANCHOR", sample_group.anchor_prompts[0]),
        *[("OPEN", p) for p in sample_group.open_prompts[: args.num_prompts - 1]],
    ]

    for label, prompt in sample_prompts[: args.num_prompts]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print("\n" + "=" * 70)
        print(f"[{label}] PROMPT:\n{prompt}")
        print("-" * 70)
        print(f"RAW COMPLETION (no extraction applied):\n{raw_completion!r}")
        print("=" * 70)

    print(
        "\nLook for literal '<think>' / '</think>' / '<answer>' / '</answer>' "
        "substrings above. If they're missing entirely, extraction will always "
        "fail regardless of what the model actually 'meant'. If they ARE "
        "present but eval still scored 0%, the bug is in extraction/judging, "
        "not generation."
    )


if __name__ == "__main__":
    main()