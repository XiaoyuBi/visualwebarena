"""Fireworks RFT evaluator for VisualWebArena web-agent action prediction.

Reward function for single-turn web agent steps.  Each dataset row contains:
  - messages: [system, (few-shot)?, user]  — prompt sent to the model
  - ground_truth: "<normalized_action>|||<0|1>"
      normalized_action  — the clean parsed action from the trajectory, e.g.
                           "click [34]", "type [5] [blue kayak]",
                           "stop [miguel_ito@example.com]"
      0 / 1              — trajectory outcome: FAIL (0) or PASS (1)
  - success: bool        — same trajectory outcome as a boolean

During RFT rollouts Fireworks appends the model-generated assistant turn as the
last message.  This function:
  1. Extracts the action from the model's CoT response (the ```action``` block).
  2. Normalises it (same way the dataset builder normalises ground_truth).
  3. Returns a score in [0.0, 1.0]:

     1.0  — exact action match
     0.3  — same action verb, different target (right intent, wrong element)
     0.1  — valid action verb but completely different from expected
     0.0  — no action block found / malformed output

The trajectory success flag packed in ground_truth (|||0/1) is available for
future reward-shaping experiments (e.g. applying a trajectory-level bonus on
top of action-match) but is not used in the current scoring formula.
"""

from __future__ import annotations

import re

from fireworks import reward_function

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches the first ```...``` block in the model's CoT response.
_ACTION_BLOCK_RE = re.compile(r"```([^`]+)```")

# ' where [id]' suffix added by the VWA HTML renderer to parsed_action strings.
_WHERE_SUFFIX_RE = re.compile(r"\s+where\s+\[\d+\]$")

# Valid web-agent action verbs (aligned with VWA action space).
_VALID_VERBS: frozenset[str] = frozenset({
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

def _extract_action(model_output: str) -> str:
    """Pull the first ```...``` block from the model's CoT response.

    Returns the first line of the block (the action itself), or "" if no
    block is found.  Inner newlines inside bracket arguments are removed
    before line-splitting so that actions like 'type [5] [text\\n]' are
    handled correctly.
    """
    m = _ACTION_BLOCK_RE.search(model_output)
    if not m:
        return ""
    content = m.group(1).strip()
    # Strip embedded newlines inside [...] before splitting on newlines.
    content = re.sub(
        r"\[([^\]]*)\]",
        lambda b: "[" + b.group(1).replace("\n", "").strip() + "]",
        content,
    )
    return content.split("\n")[0].strip()


def _normalize(action: str) -> str:
    """Canonicalise an action string for comparison.

    - Strip trailing ' where [id]' clause.
    - Remove embedded newlines and extra whitespace inside bracket arguments.
    - Strip leading/trailing whitespace.
    """
    action = _WHERE_SUFFIX_RE.sub("", action.strip())
    action = re.sub(
        r"\[([^\]]*)\]",
        lambda m: "[" + m.group(1).replace("\n", "").strip() + "]",
        action,
    )
    return action.strip()


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

@reward_function(id="vwa-action-match")
def evaluate(messages: list, ground_truth: str = "|||", **kwargs) -> dict:
    """Score a model-generated web-agent action against the ground truth.

    Args:
        messages:     Full conversation including the model's generated
                      assistant turn as the last message.
        ground_truth: "<normalized_action>|||<0|1>" from the dataset row.
        **kwargs:     Other dataset fields (ignored here).

    Returns:
        {"score": float} where score is in [0.0, 1.0].
    """
    # Unpack ground_truth — format is "<action>|||<0|1>"
    parts = ground_truth.split("|||", 1)
    expected = _normalize(parts[0])
    # traj_success = parts[1].strip() == "1" if len(parts) > 1 else False
    # (reserved for future trajectory-level reward shaping)

    if not messages:
        return {"score": 0.0}

    model_output = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    predicted = _normalize(_extract_action(model_output))

    if not predicted:
        return {"score": 0.0}

    # Exact match — full credit.
    if predicted == expected:
        return {"score": 1.0}

    pred_verb = predicted.split()[0] if predicted else ""
    exp_verb = expected.split()[0] if expected else ""

    # Same verb, different target — partial credit (right intent, wrong element).
    if pred_verb and pred_verb == exp_verb:
        return {"score": 0.3}

    # Valid verb but different action — minimal credit for correct format.
    if pred_verb in _VALID_VERBS:
        return {"score": 0.1}

    # No valid action structure.
    return {"score": 0.0}
