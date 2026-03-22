"""Per-step string_match / url_match intrinsic metric from VisualWebArena render HTML.

Reads only ``render_*.html`` files (inspect or folder mode). Parses gold ``eval`` from
the embedded ``<pre>`` block. For ``string_match`` / ``url_match`` eval types, scores
each step and reports ``if_correct_end`` / ``if_correct_somewhere``. Other eval types
yield ``\"NA\"`` for those blocks.

``fuzzy_match`` / ``ua_match`` branches in ``reference_answers`` are not reproduced
here (they require an LLM in the harness); those tasks get ``if_correct_*`` false and
``note`` explaining the skip.

Usage::

    python intrinsic_metric/step_string_url_eval.py --inspect results/shopping/shopping_gpt5mini_som_0_100/render_11.html
    python intrinsic_metric/step_string_url_eval.py results/reddit/reddit_gpt5mini_som_0_100/
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow ``python intrinsic_metric/step_string_url_eval.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nltk.tokenize import word_tokenize  # type: ignore[import]

from intrinsic_metric.vision_capability import (
    TaskConfig,
    parse_render_html,
)

METRIC_NAME = "step_string_url"


# Local copies of harness logic (avoids importing ``evaluation_harness`` → ``browser_env``).
def _clean_answer(answer: str) -> str:
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    elif answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower()


def _exact_match(ref: str, pred: str | int) -> float:
    if isinstance(pred, int):
        pred = str(pred)
    return float(_clean_answer(pred) == _clean_answer(str(ref)))


def _must_include(ref: str, pred: str) -> float:
    clean_ref = _clean_answer(ref)
    clean_pred = _clean_answer(pred)
    if len(word_tokenize(clean_ref)) == 1:
        tok_pred = word_tokenize(clean_pred)
        return float(clean_ref in tok_pred)
    return float(clean_ref in clean_pred)


def _must_exclude(ref: str, pred: str) -> float:
    clean_ref = _clean_answer(ref)
    clean_pred = _clean_answer(pred)
    if len(word_tokenize(clean_ref)) == 1:
        tok_pred = word_tokenize(clean_pred)
        return float(clean_ref not in tok_pred)
    return float(clean_ref not in clean_pred)


def _str_2_int(s: str) -> int | None:
    try:
        s = str(s).strip()
        if "," in s:
            s = s.replace(",", "")
        return int(s)
    except ValueError:
        return None


def _compare_inequality(
    value: int | float, inequality: str, tol: float = 1e-8
) -> bool:
    ops = {
        "<=": lambda x, y: x <= y + tol,
        ">=": lambda x, y: x >= y - tol,
        "==": lambda x, y: abs(x - y) <= tol,
        "<": lambda x, y: x < y + tol,
        ">": lambda x, y: x > y - tol,
    }
    for op, func in ops.items():
        if op in inequality:
            _, num = inequality.split(op)
            return func(value, float(num.strip()))
    raise ValueError(f"Invalid inequality string: {inequality}")


# ---------------------------------------------------------------------------
# Parse ``eval`` dict from render preamble (same source as RenderHelper)
# ---------------------------------------------------------------------------


def _extract_config_pre_text(html: str) -> str:
    m = re.search(r"<pre>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else ""


def parse_eval_dict_from_preamble(html: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse ``eval: {...}`` from the first config ``<pre>`` block using ``ast.literal_eval``.

    Returns ``(eval_dict, error_message)``.
    """
    pre = _extract_config_pre_text(html)
    if not pre.strip():
        return None, "no <pre> config block found"

    key = "eval:"
    idx = pre.find(key)
    if idx < 0:
        return None, "no eval: key in config pre block"

    rest = pre[idx + len(key) :].lstrip()
    if not rest.startswith("{"):
        return None, "eval value does not start with {"

    depth = 0
    end = -1
    for i, c in enumerate(rest):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None, "unbalanced braces in eval dict"

    blob = rest[:end]
    try:
        ev = ast.literal_eval(blob)
    except (ValueError, SyntaxError) as e:
        return None, f"ast.literal_eval failed: {e}"

    if not isinstance(ev, dict):
        return None, "eval is not a dict"
    return ev, None


# ---------------------------------------------------------------------------
# STOP answer extraction from render step blocks
# ---------------------------------------------------------------------------

_STOP_PARSED_RE = re.compile(
    r"^stop\s*\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL
)
_ACTION_ANSWER_RE = re.compile(
    r"""['"]answer['"]\s*:\s*(['"])(.*?)\1""", re.DOTALL
)


def extract_stop_answer(parsed_action: str, block_html: str) -> str | None:
    """Return answer string if this step is STOP, else None."""
    pa = parsed_action.strip()
    m = _STOP_PARSED_RE.match(pa)
    if m:
        return m.group(1)

    # Fallback: repr(action) in action_object
    ao = re.search(
        r"<div class='action_object'[^>]*>\s*<pre>(.*?)</pre>",
        block_html,
        re.DOTALL | re.IGNORECASE,
    )
    if not ao:
        return None
    rep = ao.group(1)
    am = _ACTION_ANSWER_RE.search(rep)
    if am:
        return am.group(2)
    return None


def _split_render_step_blocks(html: str) -> tuple[str, list[str]]:
    """Return preamble HTML and list of HTML fragments after each ``<h2>New Page</h2>``."""
    new_page = re.compile(r"<h2>New Page</h2>", re.IGNORECASE)
    parts = new_page.split(html)
    if len(parts) < 2:
        return parts[0] if parts else "", []
    return parts[0], parts[1:]


# ---------------------------------------------------------------------------
# String score (deterministic harness parity; no LLM)
# ---------------------------------------------------------------------------


def score_string_answer_no_llm(pred: str, eval_block: dict, intent: str) -> tuple[float, str | None]:
    """Mirror ``StringEvaluator`` for non-LLM approaches. Returns (score, note)."""
    del intent  # fuzzy / ua_match use intent; not used in deterministic path
    ref_answers = eval_block.get("reference_answers")
    if not isinstance(ref_answers, dict):
        return 0.0, "missing reference_answers"

    if "fuzzy_match" in ref_answers:
        return 0.0, "fuzzy_match requires LLM — not evaluated in this metric"

    pred_cur: Any = _clean_answer(pred)
    score = 1.0
    for approach, value in ref_answers.items():
        match approach:
            case "exact_match":
                score *= _exact_match(ref=value, pred=pred_cur)
            case "required_values":
                required_values = value
                assert isinstance(required_values, list)
                pred_cur = _str_2_int(str(pred_cur))
                if pred_cur is None:
                    score = 0.0
                else:
                    for v in required_values:
                        value_or = v.split(" |OR| ")
                        score *= any(
                            _compare_inequality(pred_cur, x)
                            for x in value_or
                        )
            case "must_include":
                assert isinstance(value, list)
                for must_value in value:
                    value_or = must_value.split(" |OR| ")
                    score *= any(
                        _must_include(ref=v, pred=str(pred_cur))
                        for v in value_or
                    )
            case "must_exclude":
                assert isinstance(value, list)
                for must_excl_value in value:
                    score *= _must_exclude(
                        ref=must_excl_value, pred=str(pred_cur)
                    )
            case "one_of":
                assert isinstance(value, list)
                found = False
                for one_of_value in value:
                    one_of_value = _clean_answer(one_of_value)
                    if one_of_value in pred_cur:
                        found = True
                        break
                score *= float(found)
            case "fuzzy_match":
                return 0.0, "fuzzy_match requires LLM — not evaluated in this metric"
            case _:
                continue
    return float(score), None


# ---------------------------------------------------------------------------
# URL score (per URL string)
# ---------------------------------------------------------------------------


def clean_url(url: str) -> str:
    url = str(url)
    url = url.replace("localhost", "127.0.0.1")
    if url.endswith("/"):
        url = url[:-1]
    return url


def score_url_match(pred_url: str, eval_block: dict) -> float:
    ref_raw = eval_block.get("reference_url", "")
    if not isinstance(ref_raw, str):
        return 0.0
    pred = clean_url(pred_url)
    ref_urls = ref_raw.split(" |OR| ")
    ref_urls = [clean_url(u) for u in ref_urls]
    matching_rule = eval_block.get("url_note", "EXACT")
    if matching_rule == "EXACT":
        return 1.0 if pred in ref_urls else 0.0
    if matching_rule == "GOLD in PRED":
        return 1.0 if any(ref in pred for ref in ref_urls) else 0.0
    raise ValueError(f"Unknown matching rule: {matching_rule}")


# ---------------------------------------------------------------------------
# Core evaluation for one HTML file
# ---------------------------------------------------------------------------


@dataclass
class BlockResult:
    string_match: Any  # "NA" | dict
    url_match: Any


def evaluate_render_html(html: str, _html_path: Path) -> tuple[TaskConfig, BlockResult | None, str | None]:
    """Returns (task_config, result_or_none, error)."""
    eval_dict, err = parse_eval_dict_from_preamble(html)
    config, steps = parse_render_html(html)

    if err:
        return config, None, err
    assert eval_dict is not None

    eval_types = eval_dict.get("eval_types", [])
    if not isinstance(eval_types, list):
        eval_types = []

    has_string = "string_match" in eval_types
    has_url = "url_match" in eval_types

    step_blocks = _split_render_step_blocks(html)[1]

    # Re-extract stop answers with block HTML for fallback parsing
    stop_answers: list[tuple[int, str]] = []
    for i, step in enumerate(steps):
        block = step_blocks[i] if i < len(step_blocks) else ""
        ans = extract_stop_answer(step.parsed_action, block)
        if ans is not None:
            stop_answers.append((i, ans))

    intent = config.intent

    string_out: Any = "NA"
    url_out: Any = "NA"

    if has_string:
        per_step_s: list[dict[str, Any]] = []
        note: str | None = None
        stop_scores: dict[int, float] = {}

        for i, step in enumerate(steps):
            block = step_blocks[i] if i < len(step_blocks) else ""
            ans = extract_stop_answer(step.parsed_action, block)
            if ans is None:
                per_step_s.append(
                    {"step_index": i, "if_correct": "NA"}
                )
            else:
                sc, n = score_string_answer_no_llm(ans, eval_dict, intent)
                if n:
                    note = n
                stop_scores[i] = sc
                per_step_s.append(
                    {
                        "step_index": i,
                        "if_correct": bool(sc == 1.0),
                    }
                )

        last_stop_score = 0.0
        if stop_answers:
            last_idx = stop_answers[-1][0]
            last_stop_score = stop_scores.get(last_idx, 0.0)

        any_good = any(v >= 1.0 for v in stop_scores.values())

        if_correct_end = bool(last_stop_score >= 1.0) if stop_answers else False
        if_correct_somewhere = bool(any_good)

        string_out = {
            "if_correct_end": if_correct_end,
            "if_correct_somewhere": if_correct_somewhere,
            "per_step": per_step_s,
        }
        if note:
            string_out["note"] = note

    if has_url:
        per_step_u: list[dict[str, Any]] = []
        step_scores: list[float] = []
        for i, step in enumerate(steps):
            try:
                sc = score_url_match(step.url, eval_dict)
            except ValueError as e:
                return config, None, str(e)
            step_scores.append(sc)
            per_step_u.append(
                {"step_index": i, "if_correct": bool(sc == 1.0)}
            )
        if_correct_end_u = bool(step_scores[-1] == 1.0) if step_scores else False
        if_correct_somewhere_u = any(s >= 1.0 for s in step_scores)
        url_out = {
            "if_correct_end": if_correct_end_u,
            "if_correct_somewhere": if_correct_somewhere_u,
            "per_step": per_step_u,
        }

    return config, BlockResult(string_match=string_out, url_match=url_out), None


# ---------------------------------------------------------------------------
# Aggregation (folder mode)
# ---------------------------------------------------------------------------


@dataclass
class FileSummary:
    file: str
    task_id: str
    total_steps: int
    string_match: Any
    url_match: Any
    error: str | None = None


def aggregate_folder(summaries: list[FileSummary]) -> dict[str, Any]:
    n = len(summaries)
    str_applicable = 0
    str_end = 0
    str_some = 0
    url_applicable = 0
    url_end = 0
    url_some = 0
    total_steps = 0

    for s in summaries:
        total_steps += s.total_steps
        if s.error:
            continue
        sm = s.string_match
        um = s.url_match
        if isinstance(sm, dict):
            str_applicable += 1
            if sm.get("if_correct_end"):
                str_end += 1
            if sm.get("if_correct_somewhere"):
                str_some += 1
        if isinstance(um, dict):
            url_applicable += 1
            if um.get("if_correct_end"):
                url_end += 1
            if um.get("if_correct_somewhere"):
                url_some += 1

    return {
        "num_files": n,
        "total_steps": total_steps,
        "string_match": {
            "trajectories_with_metric": str_applicable,
            "if_correct_end_count": str_end,
            "if_correct_somewhere_count": str_some,
            "if_correct_end_rate": str_end / str_applicable if str_applicable else None,
            "if_correct_somewhere_rate": str_some / str_applicable
            if str_applicable
            else None,
        },
        "url_match": {
            "trajectories_with_metric": url_applicable,
            "if_correct_end_count": url_end,
            "if_correct_somewhere_count": url_some,
            "if_correct_end_rate": url_end / url_applicable if url_applicable else None,
            "if_correct_somewhere_rate": url_some / url_applicable
            if url_applicable
            else None,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Per-step string_match / url_match metric from render_*.html trajectories."
        )
    )
    parser.add_argument(
        "--inspect",
        type=Path,
        metavar="HTML",
        default=None,
        help="Single render_*.html: print detailed JSON and save by default.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing render_*.html files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write report to this file.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write report files (only print to stdout).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        metavar="K",
        help="In folder mode, only process the first K render_*.html files.",
    )
    args = parser.parse_args(argv)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- inspect ----
    if args.inspect is not None:
        html_path = args.inspect.expanduser().resolve()
        if not html_path.is_file():
            print(f"Error: not a file: {html_path}", file=sys.stderr)
            return 1
        html = html_path.read_text(encoding="utf-8", errors="replace")
        cfg, result, err = evaluate_render_html(html, html_path)

        eval_dict, parse_err = parse_eval_dict_from_preamble(html)
        eval_types: Any = (
            eval_dict.get("eval_types", []) if eval_dict and not parse_err else []
        )

        payload: dict[str, Any] = {
            "generated": generated,
            "file": str(html_path),
            "task_id": cfg.task_id,
            "intent": cfg.intent,
            "eval_types": eval_types,
        }
        if err:
            payload["error"] = err
        elif result:
            payload["string_match"] = result.string_match
            payload["url_match"] = result.url_match
        else:
            payload["error"] = "unknown evaluation error"

        print(json.dumps(payload, ensure_ascii=False, indent=2))

        if not args.no_save:
            out_path = args.output or (
                Path(__file__).resolve().parent
                / "results"
                / f"{METRIC_NAME}_{html_path.parent.name}_{html_path.stem}.json"
            )
            out_path = out_path.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote inspect JSON to: {out_path}", file=sys.stderr)
        return 0

    # ---- folder ----
    if args.folder is None:
        parser.error(
            "the following arguments are required: folder (unless using --inspect)"
        )

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        return 1

    html_files = sorted(folder.glob("render_*.html"))
    if not html_files:
        print(f"No render_*.html files found in {folder}", file=sys.stderr)
        return 1
    if args.topk is not None:
        html_files = html_files[: args.topk]

    per_file: list[dict[str, Any]] = []
    summaries: list[FileSummary] = []

    for f in html_files:
        html = f.read_text(encoding="utf-8", errors="replace")
        cfg, steps = parse_render_html(html)
        cfg, result, err = evaluate_render_html(html, f)
        eval_dict, _ = parse_eval_dict_from_preamble(html)
        et = eval_dict.get("eval_types", []) if eval_dict else []

        entry: dict[str, Any] = {
            "file": str(f),
            "task_id": cfg.task_id,
            "total_steps": len(steps),
            "eval_types": et,
        }
        if err:
            entry["error"] = err
        elif result:
            entry["string_match"] = result.string_match
            entry["url_match"] = result.url_match

        per_file.append(entry)
        summaries.append(
            FileSummary(
                file=str(f),
                task_id=str(cfg.task_id),
                total_steps=len(steps),
                string_match=result.string_match if result else "NA",
                url_match=result.url_match if result else "NA",
                error=err,
            )
        )

    agg = aggregate_folder(summaries)

    out: dict[str, Any] = {
        "generated": generated,
        "folder": str(folder),
        "aggregation": agg,
        "per_file": per_file,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if not args.no_save:
        out_path = args.output or (
            Path(__file__).resolve().parent
            / "results"
            / f"{METRIC_NAME}_{folder.name}.json"
        )
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote JSON summary to: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
