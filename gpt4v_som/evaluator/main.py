"""Scoring logic for VisualWebArena web-agent action prediction.

Used by test_evaluator.py (the Fireworks @evaluation_test entry point) and
can also be called directly for local testing.

Each dataset row contains:
  - messages: [system, (few-shot)?, user, assistant]
      The assistant turn is appended by Fireworks during RFT rollouts.
  - ground_truth: "<normalized_action>|||<0|1>"
      normalized_action  — the clean parsed action from the trajectory
      0 / 1              — trajectory outcome: FAIL (0) or PASS (1)

Scoring (4-tier, discounted by trajectory outcome):
  1.0  — exact action match  (×0.1 if trajectory failed)
  0.3  — same action verb, different target  (×0.1 if trajectory failed)
  0.1  — valid action verb but completely different  (×0.1 if trajectory failed)
  0.0  — no action block found / malformed output
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

ACTION_BLOCK_RE = re.compile(r"```([^`]+)```")

_WHERE_SUFFIX_RE = re.compile(r"\s+where\s+\[\d+\]$")

VALID_VERBS: frozenset[str] = frozenset({
    "click",
    "type",
    "scroll",
    "hover",
    "goto",
    "go_back",
    "stop",
    "press",
    "new_tab",
    "close_tab",
    "page_focus",
    "page_close",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_action(model_output: str) -> str:
    """Pull the first ```...``` block from the model's CoT response.

    Returns the first line of the block (the action itself), or "" if no
    block is found.  Inner newlines inside bracket arguments are removed
    before line-splitting so that actions like 'type [5] [text\\n]' are
    handled correctly.
    """
    m = ACTION_BLOCK_RE.search(model_output)
    if not m:
        return ""
    content = m.group(1).strip()
    content = re.sub(
        r"\[([^\]]*)\]",
        lambda b: "[" + b.group(1).replace("\n", "").strip() + "]",
        content,
    )
    return content.split("\n")[0].strip()


def normalize(action: str) -> str:
    """Canonicalise an action string for comparison."""
    action = _WHERE_SUFFIX_RE.sub("", action.strip())
    action = re.sub(
        r"\[([^\]]*)\]",
        lambda m: "[" + m.group(1).replace("\n", "").strip() + "]",
        action,
    )
    return action.strip()


def score_action(messages: list, ground_truth: str = "|||") -> float:
    """Score a model-generated web-agent action against the ground truth.

    Args:
        messages:     Full conversation including the model's generated
                      assistant turn as the last message.
        ground_truth: "<normalized_action>|||<0|1>" from the dataset row.

    Returns:
        Score in [0.0, 1.0].
    """
    parts = ground_truth.split("|||", 1)
    expected = normalize(parts[0])
    passed = parts[1].strip() == "1" if len(parts) > 1 else True
    trajectory_multiplier = 1.0 if passed else 0.1

    if not messages:
        return 0.0

    model_output = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    predicted = normalize(extract_action(model_output))

    if not predicted:
        return 0.0

    if predicted == expected:
        return 1.0 * trajectory_multiplier

    pred_verb = predicted.split()[0] if predicted else ""
    exp_verb = expected.split()[0] if expected else ""

    if pred_verb and pred_verb == exp_verb:
        return 0.3 * trajectory_multiplier

    if pred_verb in VALID_VERBS:
        return 0.1 * trajectory_multiplier

    return 0.0
