"""
Format Reward (r_fmt), per Appendix C.1:

  "To ensure the model adheres to the Chain-of-Thought (CoT) reasoning
   requirement, we enforce a strict structural format. The format reward
   r_fmt is binary-based:
   - Structure Check: The generation must be enclosed within [reasoning]
     and [answer] tags. The reasoning process must be non-empty, and the
     final answer must be extracted successfully.
   - Math-Specific Constraint: for math tasks, boxed{...} inside answer tags.
   If the generation satisfies all formatting constraints, r_fmt = 1.25;
   otherwise, a penalty of r_fmt = -1.0 is applied."
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from config import CONFIG


class BaseReward(ABC):
    @abstractmethod
    def score(self, prompt: str, response: str, **kwargs) -> float:
        raise NotImplementedError


def _extract_tagged_content(response: str, tags: tuple[str, ...]) -> str | None:
    """
    Returns the content strictly between an opening and closing tag, or
    None if the tags are missing, reversed, or empty between them.
    """
    open_tag, close_tag = tags[0], tags[-1]
    start = response.find(open_tag)
    end = response.find(close_tag)

    if start == -1 or end == -1 or start >= end:
        return None

    content = response[start + len(open_tag): end].strip()
    return content or None  # empty-after-strip counts as missing


def extract_reasoning_content(response: str, reasoning_tags: tuple = CONFIG.reward.reasoning_tags) -> str | None:
    return _extract_tagged_content(response, reasoning_tags)


def extract_answer_content(response: str, answer_tags: tuple = CONFIG.reward.answer_tags) -> str | None:
    return _extract_tagged_content(response, answer_tags)


class FormatReward(BaseReward):
    """
    Applies to ALL prompts (anchor and open alike): checks that reasoning
    is non-empty AND the final answer is extractable. Does not itself
    grade whether the extracted answer is correct, that's r_task's job
    (see safety_task_reward.py).
    """

    def __init__(
        self,
        reasoning_tags: tuple = CONFIG.reward.reasoning_tags,
        answer_tags: tuple = CONFIG.reward.answer_tags,
        reward_value: float = CONFIG.reward.format_reward_value,
        penalty: float = CONFIG.reward.format_reward_penalty,
    ) -> None:
        self._reasoning_tags = reasoning_tags
        self._answer_tags = answer_tags
        self._reward_value = reward_value
        self._penalty = penalty

    def score(self, prompt: str, response: str, **kwargs) -> float:
        reasoning = extract_reasoning_content(response, self._reasoning_tags)
        answer = extract_answer_content(response, self._answer_tags)

        if reasoning is not None and answer is not None:
            return self._reward_value
        return self._penalty