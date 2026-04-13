#!/usr/bin/env python3
"""Build Fireworks AI RFT JSONL from VisualWebArena render_*.html trajectories.

Each line is one agent step in prompt-only format (no assistant turn), suitable
for Reinforcement Fine-Tuning (RFT) per:
https://docs.fireworks.ai/fine-tuning/how-rft-works

Key differences from create_sft_dataset.py (SFT):
  - messages = [system, (few-shot)?, user]   — NO assistant turn
  - skip step when parsed_action is empty    — not raw_prediction
  - ground_truth = "<normalized_action>|||<0|1>"
      normalized_action: parsed_action with ' where [id]' suffix stripped,
                         inner newlines inside brackets removed
      0|1: trajectory FAIL (0) or PASS (1) from results.txt
  - success field kept at root for reference

ground_truth examples:
  "click [34]|||0"
  "stop [miguel_ito@example.com]|||1"
  "type [5] [blue kayak]|||0"

Usage (from repo root)::

    python gpt4v_som/create_rl_dataset.py \\
        --raw-dir gpt4v_som/raw \\
        -o gpt4v_som/dataset/rl_image_resized_noex.jsonl \\
        --no-examples --image-resize 720
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

# Import all shared HTML-parsing and image-handling utilities from create_sft_dataset.
# Both scripts live in gpt4v_som/ and are run from the repo root, so the import works
# when the cwd is the repo root (sys.path includes '.').
import os
sys.path.insert(0, os.path.dirname(__file__))

from create_sft_dataset import (
    DEFAULT_PROMPT,
    DEFAULT_RAW,
    TaskConfig,
    StepData,
    _few_shot_messages,
    _truncate,
    _user_message_from_template,
    _load_results,
    parse_render_html,
    render_id_from_filename,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUT = _REPO_ROOT / "gpt4v_som/dataset/rl_trajectories.jsonl"

# ---------------------------------------------------------------------------
# Action normalisation
# ---------------------------------------------------------------------------

_WHERE_SUFFIX_RE = re.compile(r"\s+where\s+\[\d+\](\s*)$")


def _normalize_action(action: str) -> str:
    """Canonicalise a parsed_action string for use as ground_truth.

    1. Strip trailing ' where [id]' clause added by the HTML renderer.
    2. Remove embedded newlines inside bracket arguments (e.g. type puts \\n
       inside the content bracket: 'type [5] [blue kayak\\n]').
    3. Strip leading/trailing whitespace.
    """
    action = _WHERE_SUFFIX_RE.sub("", action.strip())
    # Remove \n inside [...] argument brackets
    action = re.sub(r"\[([^\]]*)\]", lambda m: "[" + m.group(1).replace("\n", "").strip() + "]", action)
    return action.strip()


# ---------------------------------------------------------------------------
# RFT step-prompt builder
# ---------------------------------------------------------------------------

def build_rl_step_prompts(
    config: TaskConfig,
    steps: list[StepData],
    prompt_data: dict[str, Any],
    *,
    success: bool | None,
    include_examples: bool,
    max_obs_chars: int,
    repo_root: Path,
    ignore_images: bool = False,
    resize_max: int | None = None,
) -> list[dict[str, Any]]:
    """One JSON object per usable step: prompt-only messages + ground_truth.

    Each object is a single-turn RFT prompt: the model sees
    [system intro] + [few-shot turns] + [one user message] and must generate
    the action.  The assistant turn is intentionally omitted — Fireworks
    generates it as a rollout during RFT training.

    Steps are skipped when parsed_action is empty (action could not be parsed).
    """
    intent = html.unescape(config.intent.strip())
    if not intent:
        return []

    intro = html.unescape(prompt_data.get("intro", ""))
    template = prompt_data.get(
        "template",
        "OBSERVATION: {observation}\nURL: {url}\nOBJECTIVE: {objective}\n"
        "PREVIOUS ACTION: {previous_action}",
    )

    prefix: list[dict[str, Any]] = [{"role": "system", "content": intro}]
    if include_examples:
        examples = prompt_data.get("examples") or []
        prefix.extend(
            _few_shot_messages(
                examples,
                repo_root,
                ignore_images=ignore_images,
                resize_max=resize_max,
            )
        )

    success_flag = "1" if success is True else "0"

    out: list[dict[str, Any]] = []
    for step in steps:
        parsed = step.parsed_action.strip()
        if not parsed:
            continue

        obs = html.unescape(step.text_obs.strip())
        obs = _truncate(obs, max_obs_chars)
        url = html.unescape(step.url.strip())
        prev = html.unescape(step.prev_action.strip())

        user_msg = _user_message_from_template(
            template,
            observation=obs,
            url=url,
            objective=intent,
            previous_action=prev,
            screenshot_b64=step.screenshot_b64,
            ignore_images=ignore_images,
            resize_max=resize_max,
        )

        messages: list[dict[str, Any]] = list(prefix) + [user_msg]
        ground_truth = _normalize_action(parsed) + "|||" + success_flag

        out.append({
            "messages": messages,
            "ground_truth": ground_truth,
            "success": success,
            "_step_index": step.step_index,
        })

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert render_*.html trajectories to Fireworks RFT JSONL "
            "(prompt-only, with ground_truth=action|||0/1)."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW,
        help="Directory containing subfolders with render_*.html (default: gpt4v_som/raw)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT,
        help="Prompt JSON (intro, examples, template)",
    )
    parser.add_argument(
        "--max-obs-chars",
        type=int,
        default=15360,
        help="Max characters for accessibility-tree text per step (default: 15360)",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Omit few-shot example turns from each prompt",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        metavar="K",
        help="Process at most K HTML files total (for testing)",
    )
    parser.add_argument(
        "--min-render-id",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Only include render_<id>.html with numeric id >= N (default: 100). "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-render-id",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only include render_<id>.html with numeric id < N (exclusive). "
            "Use with --min-render-id 0 for eval slice, e.g. idx < 100."
        ),
    )
    parser.add_argument(
        "--ignore-images",
        action="store_true",
        help="Replace every screenshot with a 1x1 PNG placeholder.",
    )
    parser.add_argument(
        "--image-resize",
        type=int,
        default=None,
        metavar="MAX_SIDE",
        help=(
            "Resize screenshots so the longest side is at most MAX_SIDE pixels. "
            "Aspect ratio preserved. Requires Pillow. "
            "Ignored when --ignore-images is set."
        ),
    )
    parser.add_argument(
        "--with-meta",
        action="store_true",
        help="Add meta (source path, task_id, domain, step_index) to each JSON line",
    )
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()

    if not prompt_path.is_file():
        print(f"Error: prompt file not found: {prompt_path}", file=sys.stderr)
        return 1
    if not raw_dir.is_dir():
        print(f"Error: raw directory not found: {raw_dir}", file=sys.stderr)
        return 1

    prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))

    subdirs: list[Path] = sorted(sub for sub in raw_dir.iterdir() if sub.is_dir())
    html_files: list[Path] = []
    for sub in subdirs:
        for f in sorted(sub.glob("render_*.html")):
            html_files.append(f)

    mn = args.min_render_id
    if mn > 0:
        html_files = [
            f for f in html_files
            if (rid := render_id_from_filename(f)) is not None and rid >= mn
        ]

    if args.max_render_id is not None:
        mx = args.max_render_id
        html_files = [
            f for f in html_files
            if (rid := render_id_from_filename(f)) is not None and rid < mx
        ]

    if args.topk is not None:
        html_files = html_files[: args.topk]

    results_cache: dict[Path, dict[int, bool]] = {}
    for f in html_files:
        sub = f.parent
        if sub not in results_cache:
            results_cache[sub] = _load_results(sub)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for html_path in html_files:
            rid = render_id_from_filename(html_path)
            sub_results = results_cache.get(html_path.parent, {})
            success: bool | None = sub_results.get(rid) if rid is not None else None

            try:
                text = html_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                print(f"Warning: skip {html_path}: {e}", file=sys.stderr)
                skipped += 1
                continue

            config, steps = parse_render_html(text)
            resize_max = args.image_resize if not args.ignore_images else None

            step_prompts = build_rl_step_prompts(
                config,
                steps,
                prompt_data,
                success=success,
                include_examples=not args.no_examples,
                max_obs_chars=args.max_obs_chars,
                repo_root=_REPO_ROOT,
                ignore_images=args.ignore_images,
                resize_max=resize_max,
            )

            if not step_prompts:
                skipped += 1
                continue

            for prompt in step_prompts:
                step_index = prompt.pop("_step_index")
                line_obj: dict[str, Any] = dict(prompt)
                if args.with_meta:
                    line_obj["meta"] = {
                        "source_html": str(html_path.relative_to(_REPO_ROOT)),
                        "task_id": config.task_id,
                        "domain": html_path.parent.name,
                        "step_index": step_index,
                    }
                out_f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                written += 1

    print(
        f"Wrote {written} steps to {out_path} (skipped {skipped} HTML files).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
