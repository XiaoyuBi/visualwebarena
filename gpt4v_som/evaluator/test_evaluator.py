"""Fireworks RFT evaluator for VisualWebArena web-agent action prediction.

This file is the pytest entry point that Fireworks discovers via
``pytest --collect``.  The actual scoring logic lives in ``main.py``.
"""

import os
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message
from eval_protocol.pytest import evaluation_test

from main import score_action

_SAMPLE = os.path.join(os.path.dirname(__file__), "sample.jsonl")


def _dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    out: List[EvaluationRow] = []
    for row in rows:
        msgs = [Message(**m) for m in row.get("messages", [])]
        out.append(
            EvaluationRow(
                messages=msgs,
                ground_truth=row.get("ground_truth", "|||"),
            )
        )
    return out


@evaluation_test(
    input_dataset=[_SAMPLE],
    dataset_adapter=_dataset_adapter,
    mode="pointwise",
)
def test_evaluate(row: EvaluationRow) -> EvaluationRow:
    messages = [
        m.model_dump() if hasattr(m, "model_dump") else m
        for m in (row.messages or [])
    ]
    ground_truth = row.ground_truth or "|||"

    score = score_action(messages, ground_truth)

    row.evaluation_result = EvaluateResult(score=score)
    return row
