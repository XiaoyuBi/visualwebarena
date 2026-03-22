"""Evaluate vision capability of a web agent from VisualWebArena render HTML trajectories.

For each step in a trajectory, uses GPT-5.4 as a multimodal judge to classify the
agent's vision capability as GOOD, BAD, or NA.

- GOOD: The step requires vision and the agent demonstrates correct visual understanding.
- BAD:  The step requires vision but the agent's visual understanding is incorrect/absent.
- NA:   The step does not require vision -- non-visual signals suffice (e.g. a11y tree \
when present); when the tree is empty, NA only if the step still does not require \
pixel-level understanding.

Usage examples::

    # Inspect a single trajectory (detailed per-step JSON)
    python intrinsic_metric/vision_capability.py --inspect results/shopping/shopping_gpt5mini_som_0_100/render_11.html

    # Evaluate an entire folder of trajectories
    python intrinsic_metric/vision_capability.py results/reddit/reddit_gpt5mini_som_0_100/

    # Evaluate only the first 5 files in a folder
    python intrinsic_metric/vision_capability.py results/reddit/reddit_gpt5mini_som_0_100/ --topk 5

Environment: set EVAL_OPENAI_API_KEY (or OPENAI_API_KEY) before running.

**Two kinds of images** enter the judge prompt (assembled in ``build_judge_messages``):

1. **Task reference image (optional):** from ``image`` in the render config, loaded
   once from disk and **reused at every step** when present.
2. **Per-step page screenshot (required for judging):** embedded in the HTML for
   each step, **different every step** — this is the step's visual **input** the
   metric evaluates against, along with the accessibility tree and other text **when
   present** (pure image tasks may omit usable text).

If a step has no screenshot in the HTML, that step is scored NA and the judge is
not called for it.

**API cost:** Images sent to the judge are downscaled to max width 720px (aspect
ratio preserved) to lower billed vision/input tokens. User message parts are ordered
so the **same task-level prefix** is reused across steps, which lets the provider
charge **cached input** rates for that prefix when applicable. Verdict JSON under
``--cache-dir`` skips paying for a **second** judge API call when step content
unchanged. Nothing here optimizes local CPU, disk, or wall time beyond that goal.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI
from PIL import Image

METRIC_NAME = "vision_capability"

# Max width for judge-bound images — lowers **API** billed vision/input tokens.
JUDGE_IMAGE_MAX_WIDTH = 720

# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

_NEW_PAGE_RE = re.compile(r"<h2>New Page</h2>", re.IGNORECASE)
_CONFIG_PRE_RE = re.compile(r"<pre>(.*?)</pre>", re.DOTALL)
_SCREENSHOT_RE = re.compile(
    r"<img src='data:image/png;base64,([^']+)'", re.IGNORECASE
)
_STATE_OBV_RE = re.compile(
    r"<div class='state_obv'><pre>(.*?)</pre>", re.DOTALL
)
_RAW_PREDICTION_RE = re.compile(
    r"<div class='raw_parsed_prediction'[^>]*><pre>(.*?)</pre></div>", re.DOTALL
)
_PARSED_ACTION_RE = re.compile(
    r"<div class='parsed_action'[^>]*><pre>(.*?)</pre></div>", re.DOTALL
)
_PREV_ACTION_RE = re.compile(
    r"<div class='prev_action'[^>]*>(.*?)</div>", re.DOTALL
)
_URL_RE = re.compile(
    r"<h3 class='url'><a href=[^>]+>URL:\s*(.*?)</a></h3>", re.IGNORECASE
)


@dataclass
class TaskConfig:
    intent: str
    image_path: str | None
    task_id: str
    comments: str
    raw_config: str


@dataclass
class StepData:
    step_index: int
    url: str
    text_obs: str
    screenshot_b64: str | None
    prev_action: str
    raw_prediction: str
    parsed_action: str


def parse_config(config_text: str) -> TaskConfig:
    """Parse the config ``<pre>`` block into a TaskConfig."""
    fields: dict[str, str] = {}
    for line in config_text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

    image_raw = fields.get("image", "None")
    if image_raw in ("None", "", "[]"):
        image_path = None
    elif image_raw.startswith("[") and image_raw.endswith("]"):
        inner = image_raw[1:-1].strip().strip("'\"")
        image_path = inner if inner else None
    else:
        image_path = image_raw

    return TaskConfig(
        intent=fields.get("intent", ""),
        image_path=image_path,
        task_id=fields.get("task_id", "unknown"),
        comments=fields.get("comments", ""),
        raw_config=config_text.strip(),
    )


def parse_render_html(html: str) -> tuple[TaskConfig, list[StepData]]:
    """Parse a render HTML file into config + ordered list of per-step data."""
    parts = _NEW_PAGE_RE.split(html)
    if len(parts) < 2:
        config_match = _CONFIG_PRE_RE.search(parts[0] if parts else "")
        config = parse_config(config_match.group(1) if config_match else "")
        return config, []

    preamble = parts[0]
    config_match = _CONFIG_PRE_RE.search(preamble)
    config = parse_config(config_match.group(1) if config_match else "")

    steps: list[StepData] = []
    for i, block in enumerate(parts[1:]):
        url_m = _URL_RE.search(block)
        state_m = _STATE_OBV_RE.search(block)
        screenshot_m = _SCREENSHOT_RE.search(block)
        raw_pred_m = _RAW_PREDICTION_RE.search(block)
        parsed_m = _PARSED_ACTION_RE.search(block)
        prev_m = _PREV_ACTION_RE.search(block)

        steps.append(
            StepData(
                step_index=i,
                url=url_m.group(1).strip() if url_m else "",
                text_obs=state_m.group(1).strip() if state_m else "",
                screenshot_b64=screenshot_m.group(1) if screenshot_m else None,
                prev_action=prev_m.group(1).strip() if prev_m else "",
                raw_prediction=raw_pred_m.group(1).strip() if raw_pred_m else "",
                parsed_action=parsed_m.group(1).strip() if parsed_m else "",
            )
        )

    return config, steps


# ---------------------------------------------------------------------------
# GPT-5.4 judge prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert evaluator assessing a web browsing agent's **vision capability** \
at a single step of a web task.

You are given:
1. The overall task the agent is trying to accomplish.
2. Optionally, a fixed task reference image from the task definition (if the task includes one).
3. The **per-step screenshot** of the current web page the agent sees — this is the primary visual state for the step and may differ from step to step.
4. Optionally, the text-based observation (accessibility tree) available to the agent. It may be missing or empty in **pure image input** tasks.
5. The agent's reasoning and final action for this step.

Your job is to judge whether this step **requires vision capability** and, if so, \
whether the agent **correctly used visual understanding**.

## Evaluation criteria

- **NA**: This step does NOT require vision. When an accessibility tree is present \
and sufficient, the correct action follows from that text (plus URL, task wording, \
etc.) alone — e.g. clicking a link by its label, typing in a named field, scrolling, \
or navigating by URL. When the tree is **absent or empty**, NA is rare: use it only \
if the step still does not require interpreting pixels in the screenshot; otherwise \
prefer GOOD or BAD.

- **GOOD**: This step REQUIRES vision capability and the agent demonstrates \
**correct** visual understanding. Examples: correctly identifying an item by its \
visual appearance in a screenshot, reading text that is only visible in the image \
but not in the accessibility tree, understanding spatial layout or visual cues \
(colors, positions, images) to make the right decision. Especially relevant when \
the accessibility tree is missing or uninformative.

- **BAD**: This step REQUIRES vision capability but the agent demonstrates \
**incorrect or absent** visual understanding. Examples: clicking the wrong image, \
misidentifying a visual element, failing to notice visual cues that should inform \
the action, or making an action that contradicts what the screenshot shows.

## Rules
- Output EXACTLY one of: GOOD, BAD, or NA.
- First provide a brief reasoning (1-2 sentences), then output your verdict on a \
new line in the format: **Verdict: GOOD**, **Verdict: BAD**, or **Verdict: NA**.
- Focus only on vision capability. Do not judge the overall task strategy or \
whether the action is optimal for other reasons.
- If there is no **per-step page screenshot** available for this step, output NA.
"""

MAX_TEXT_OBS_CHARS = 4000
# Second attempt after judge hits output/token limits (see ``_judge_output_limit_error``).
MAX_TEXT_OBS_CHARS_RETRY = MAX_TEXT_OBS_CHARS // 2


def _truncate_text_obs(
    text_obs: str, max_chars: int | None = None
) -> str:
    """Truncate long accessibility-tree text while keeping head and tail."""
    cap = max_chars if max_chars is not None else MAX_TEXT_OBS_CHARS
    if len(text_obs) <= cap:
        return text_obs
    half = cap // 2
    return text_obs[:half] + "\n... [truncated] ...\n" + text_obs[-half:]


def _judge_output_limit_error(exc: BaseException) -> bool:
    """True when the API failed because completion hit max_tokens / output limit."""
    s = str(exc).lower()
    return (
        "max_tokens" in s
        or "output limit" in s
        or "max_completion" in s
        or "could not finish the message" in s
    )


@dataclass
class StepVerdict:
    step_index: int
    verdict: str  # GOOD, BAD, or NA
    reasoning: str


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resize_image_b64_to_max_width(b64: str, max_width: int = JUDGE_IMAGE_MAX_WIDTH) -> str:
    """Downscale wide images to reduce **API** vision/input tokens; else unchanged.

    Returns base64-encoded PNG suitable for ``data:image/png;base64,...``.
    """
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        try:
            raw = base64.b64decode(b64)
        except Exception:
            logging.warning("Invalid base64 for image resize; using original.")
            return b64
    try:
        buf = io.BytesIO(raw)
        with Image.open(buf) as im:
            w, h = im.size
            if w <= max_width:
                return b64
            new_h = max(1, int(round(h * max_width / w)))
            resized = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            resized.save(out, format="PNG", optimize=True)
            return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception as e:
        logging.warning("Image resize failed; using original: %s", e)
        return b64


def _load_task_image_b64(image_path: str) -> str | None:
    """Load the optional task **reference** image from disk (base64-encoded)."""
    p = Path(image_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.is_file():
        logging.warning("Task input image not found: %s", p)
        return None
    raw = p.read_bytes()
    return base64.b64encode(raw).decode("ascii")


def build_judge_messages(
    config: TaskConfig,
    step: StepData,
    task_reference_image_b64: str | None = None,
    text_obs_max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Build the OpenAI messages list for a single judge call.

    *task_reference_image_b64* is the optional fixed image from disk (same every
    step). *step.screenshot_b64* is the per-step page screenshot from the render HTML.
    If *step.text_obs* is empty after stripping, the user message states that the
    accessibility tree was omitted (pure image input is possible).

    **API cost:** ``content`` parts are ordered so the trajectory-constant prefix
    (static task text + optional reference image) is identical on every step, which
    enables cheaper **cached** input pricing on that prefix when the provider
    applies it. Images are resized to ``JUDGE_IMAGE_MAX_WIDTH`` before sending to
    reduce billed vision tokens.

    *text_obs_max_chars* caps the accessibility-tree block (default
    ``MAX_TEXT_OBS_CHARS``); lower values are used when retrying after output-limit
    errors in ``judge_step`` / ``async_judge_step``.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    user_content: list[dict[str, Any]] = []

    # --- Static per-trajectory prefix (same every step → API cached-prefix pricing) ---
    static_task_text = f"## Overall Task\n{config.intent}\n"
    if config.comments:
        static_task_text += f"**Task comments:** {config.comments}\n"
    if task_reference_image_b64:
        static_task_text += (
            "(This task includes a fixed reference image from the task definition, "
            "shown below before the step-specific section and per-step screenshot.)\n"
        )
    user_content.append({"type": "text", "text": static_task_text})

    if task_reference_image_b64:
        ref_b64 = _resize_image_b64_to_max_width(task_reference_image_b64)
        user_content.append(
            {"type": "text", "text": "Task reference image (from disk, same every step):"}
        )
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{ref_b64}"
                },
            }
        )

    # --- Step-specific suffix (varies each step) ---
    obs = step.text_obs.strip()
    tobs_cap = (
        text_obs_max_chars
        if text_obs_max_chars is not None
        else MAX_TEXT_OBS_CHARS
    )
    if obs:
        obs_block = (
            f"**Text Observation (Accessibility Tree):**\n"
            f"```\n{_truncate_text_obs(step.text_obs, tobs_cap)}\n```\n\n"
        )
    else:
        obs_block = (
            "**Text Observation (Accessibility Tree):** *(none — empty or omitted; "
            "pure image input is possible.)*\n\n"
        )

    step_text = (
        f"## Current Step {step.step_index}\n"
        f"**URL:** {step.url}\n\n"
        f"{obs_block}"
        f"**Previous Action:** {step.prev_action}\n\n"
        f"**Agent's Reasoning and Action:**\n{step.raw_prediction}\n\n"
        f"**Parsed Action:** {step.parsed_action}\n"
    )
    user_content.append({"type": "text", "text": step_text})

    if step.screenshot_b64:
        shot_b64 = _resize_image_b64_to_max_width(step.screenshot_b64)
        user_content.append(
            {
                "type": "text",
                "text": "Per-step page screenshot (visual input for this step, from render HTML):",
            }
        )
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{shot_b64}"
                },
            }
        )

    messages.append({"role": "user", "content": user_content})
    return messages


_VERDICT_RE = re.compile(r"\*\*Verdict:\s*(GOOD|BAD|NA)\*\*", re.IGNORECASE)


def parse_verdict(response: str) -> tuple[str, str]:
    """Extract ``(verdict, reasoning)`` from the judge response text."""
    match = _VERDICT_RE.search(response)
    if match:
        verdict = match.group(1).upper()
        reasoning = response[: match.start()].strip()
        return verdict, reasoning

    upper = response.strip().upper()
    for token in ("GOOD", "BAD", "NA"):
        if token in upper:
            return token, response.strip()

    logging.warning("Could not parse verdict from response: %s", response[:200])
    return "NA", response.strip()


# ---------------------------------------------------------------------------
# OpenAI client + verdict store (skip duplicate judge **API** charges only)
# ---------------------------------------------------------------------------
# ``--cache-dir`` JSON: if a step was already judged for the same content, do not
# call the API again.
# Message layout in ``build_judge_messages``: shared task prefix before step-specific
# text + screenshot so the provider can bill **cached** rates on that prefix (API $).


def _get_api_key() -> str:
    api_key = os.environ.get("EVAL_OPENAI_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not api_key:
        raise ValueError(
            "Set EVAL_OPENAI_API_KEY or OPENAI_API_KEY environment variable."
        )
    return api_key


def _make_client() -> OpenAI:
    return OpenAI(api_key=_get_api_key(), base_url="https://api.openai.com/v1")


def _make_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=_get_api_key(), base_url="https://api.openai.com/v1")


def _content_hash(step: StepData) -> str:
    """Hash step inputs so unchanged steps skip a repeat judge API call."""
    h = hashlib.sha256()
    h.update(step.text_obs.encode("utf-8", errors="replace"))
    h.update(step.raw_prediction.encode("utf-8", errors="replace"))
    h.update(step.parsed_action.encode("utf-8", errors="replace"))
    if step.screenshot_b64:
        h.update(step.screenshot_b64[:2000].encode("ascii"))
    return h.hexdigest()[:16]


def _cache_key(html_path: Path, step: StepData) -> str:
    path_hash = hashlib.sha256(
        str(html_path.resolve()).encode()
    ).hexdigest()[:12]
    return f"{path_hash}__{html_path.name}__step{step.step_index}__{_content_hash(step)}"


def load_cache(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Load verdict JSON so re-runs can avoid paying for duplicate judge API calls."""
    cache: dict[str, dict[str, Any]] = {}
    if not cache_dir.is_dir():
        return cache
    for f in cache_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "cache_key" in data:
                cache[data["cache_key"]] = data
        except Exception:
            continue
    return cache


def save_cache_entry(
    cache_dir: Path, key: str, verdict: StepVerdict
) -> None:
    """Persist a verdict so the same step content need not invoke the API again."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "cache_key": key,
        "step_index": verdict.step_index,
        "verdict": verdict.verdict,
        "reasoning": verdict.reasoning,
    }
    (cache_dir / f"{key}.json").write_text(
        json.dumps(entry, ensure_ascii=False), encoding="utf-8"
    )


def judge_step(
    client: OpenAI,
    model: str,
    config: TaskConfig,
    step: StepData,
    task_reference_image_b64: str | None = None,
) -> StepVerdict:
    """Call the judge model to evaluate a single step's vision capability.

    On API output-limit errors, retries once with half the a11y-tree char cap; then NA.
    """
    if not step.screenshot_b64:
        return StepVerdict(
            step.step_index,
            "NA",
            "No per-step page screenshot in render HTML for this step.",
        )

    for attempt in (1, 2):
        cap = (
            MAX_TEXT_OBS_CHARS
            if attempt == 1
            else MAX_TEXT_OBS_CHARS_RETRY
        )
        messages = build_judge_messages(
            config, step, task_reference_image_b64, text_obs_max_chars=cap
        )
        try:
            response = client.chat.completions.create(
                model=model,
                reasoning_effort="medium",
                max_completion_tokens=1024,
                messages=messages,
            )
        except Exception as e:
            if not _judge_output_limit_error(e):
                raise
            if attempt == 1:
                logging.warning(
                    "Judge hit output limit (step %s); retrying with shorter "
                    "text observation (%s chars).",
                    step.step_index,
                    MAX_TEXT_OBS_CHARS_RETRY,
                )
                continue
            return StepVerdict(
                step.step_index,
                "NA",
                "Judge API output limit after truncation retry; scored NA.",
            )
        text = response.choices[0].message.content or ""
        verdict, reasoning = parse_verdict(text)
        return StepVerdict(step.step_index, verdict, reasoning)

    return StepVerdict(
        step.step_index,
        "NA",
        "Judge API output limit after truncation retry; scored NA.",
    )


def evaluate_file(
    html_path: Path,
    client: OpenAI | None,
    model: str,
    cache_dir: Path,
    cache: dict[str, dict[str, Any]],
) -> tuple[TaskConfig, list[StepVerdict]]:
    """Evaluate every step in one render HTML file.

    Uses *cache* to skip judge **API** calls when a step's verdict is already stored.

    If no step has a per-step screenshot, all steps are NA — no judge API calls.
    """
    html = html_path.read_text(encoding="utf-8", errors="replace")
    config, steps = parse_render_html(html)

    has_any_screenshot = any(s.screenshot_b64 for s in steps)
    if not has_any_screenshot:
        verdicts = [
            StepVerdict(s.step_index, "NA", "No screenshots in this trajectory.")
            for s in steps
        ]
        return config, verdicts

    task_reference_image_b64: str | None = None
    if config.image_path:
        task_reference_image_b64 = _load_task_image_b64(config.image_path)

    verdicts: list[StepVerdict] = []
    for step in steps:
        key = _cache_key(html_path, step)
        if key in cache:
            c = cache[key]
            verdicts.append(
                StepVerdict(c["step_index"], c["verdict"], c["reasoning"])
            )
            continue

        assert client is not None, (
            "OpenAI client required for steps with screenshots"
        )
        v = judge_step(
            client, model, config, step, task_reference_image_b64
        )
        verdicts.append(v)
        save_cache_entry(cache_dir, key, v)
        cache[key] = {
            "cache_key": key,
            "step_index": v.step_index,
            "verdict": v.verdict,
            "reasoning": v.reasoning,
        }

    return config, verdicts


# ---------------------------------------------------------------------------
# Async variants (for concurrent folder evaluation)
# ---------------------------------------------------------------------------


async def async_judge_step(
    aclient: AsyncOpenAI,
    model: str,
    config: TaskConfig,
    step: StepData,
    task_reference_image_b64: str | None = None,
) -> StepVerdict:
    """Async version of :func:`judge_step`."""
    if not step.screenshot_b64:
        return StepVerdict(
            step.step_index,
            "NA",
            "No per-step page screenshot in render HTML for this step.",
        )

    for attempt in (1, 2):
        cap = (
            MAX_TEXT_OBS_CHARS
            if attempt == 1
            else MAX_TEXT_OBS_CHARS_RETRY
        )
        messages = build_judge_messages(
            config, step, task_reference_image_b64, text_obs_max_chars=cap
        )
        try:
            response = await aclient.chat.completions.create(
                model=model,
                reasoning_effort="medium",
                max_completion_tokens=1024,
                messages=messages,
            )
        except Exception as e:
            if not _judge_output_limit_error(e):
                raise
            if attempt == 1:
                logging.warning(
                    "Judge hit output limit (step %s); retrying with shorter "
                    "text observation (%s chars).",
                    step.step_index,
                    MAX_TEXT_OBS_CHARS_RETRY,
                )
                continue
            return StepVerdict(
                step.step_index,
                "NA",
                "Judge API output limit after truncation retry; scored NA.",
            )
        text = response.choices[0].message.content or ""
        verdict, reasoning = parse_verdict(text)
        return StepVerdict(step.step_index, verdict, reasoning)

    return StepVerdict(
        step.step_index,
        "NA",
        "Judge API output limit after truncation retry; scored NA.",
    )


async def async_evaluate_file(
    html_path: Path,
    aclient: AsyncOpenAI,
    model: str,
    cache_dir: Path,
    cache: dict[str, dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> tuple[TaskConfig, list[StepVerdict]]:
    """Async version of :func:`evaluate_file` (same API-cost verdict cache).

    The *semaphore* limits concurrent files only (not an API-cost feature).
    """
    async with semaphore:
        logging.info("Processing %s ...", html_path.name)
        html = html_path.read_text(encoding="utf-8", errors="replace")
        config, steps = parse_render_html(html)

        has_any_screenshot = any(s.screenshot_b64 for s in steps)
        if not has_any_screenshot:
            verdicts = [
                StepVerdict(
                    s.step_index, "NA", "No screenshots in this trajectory."
                )
                for s in steps
            ]
            return config, verdicts

        task_reference_image_b64: str | None = None
        if config.image_path:
            task_reference_image_b64 = _load_task_image_b64(config.image_path)

        verdicts: list[StepVerdict] = []
        for step in steps:
            key = _cache_key(html_path, step)
            if key in cache:
                c = cache[key]
                verdicts.append(
                    StepVerdict(
                        c["step_index"], c["verdict"], c["reasoning"]
                    )
                )
                continue

            v = await async_judge_step(
                aclient, model, config, step, task_reference_image_b64
            )
            verdicts.append(v)
            save_cache_entry(cache_dir, key, v)
            cache[key] = {
                "cache_key": key,
                "step_index": v.step_index,
                "verdict": v.verdict,
                "reasoning": v.reasoning,
            }

        return config, verdicts


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryResult:
    file_path: Path
    total_steps: int
    vision_good: int
    vision_bad: int
    vision_na: int
    step_verdicts: list[StepVerdict]


def aggregate_verdicts(
    file_path: Path, verdicts: list[StepVerdict]
) -> TrajectoryResult:
    good = sum(1 for v in verdicts if v.verdict == "GOOD")
    bad = sum(1 for v in verdicts if v.verdict == "BAD")
    na = sum(1 for v in verdicts if v.verdict == "NA")
    total = len(verdicts)
    assert good + bad + na == total, (
        f"Counts don't sum: {good}+{bad}+{na} != {total}"
    )
    return TrajectoryResult(file_path, total, good, bad, na, verdicts)


@dataclass
class FolderResult:
    total_steps: int
    vision_good: int
    vision_bad: int
    vision_na: int
    num_files: int
    per_file: list[TrajectoryResult]


def aggregate_folder(results: list[TrajectoryResult]) -> FolderResult:
    return FolderResult(
        total_steps=sum(r.total_steps for r in results),
        vision_good=sum(r.vision_good for r in results),
        vision_bad=sum(r.vision_bad for r in results),
        vision_na=sum(r.vision_na for r in results),
        num_files=len(results),
        per_file=results,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate web agent vision capability from render_*.html files "
            "using GPT-5.4 as a multimodal judge."
        )
    )
    parser.add_argument(
        "--inspect",
        type=Path,
        metavar="HTML",
        default=None,
        help=(
            "Single render_*.html file: evaluate and print detailed report. "
            "Default save: intrinsic_metric/results/"
            f"{METRIC_NAME}_<parent_dir>_<render_stem>.json "
            "(e.g. …_shopping_gpt5mini_som_0_100_render_1.json), unlike folder mode "
            f"which omits the render stem ({METRIC_NAME}_<folder>.json)."
        ),
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing render_*.html files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="Judge model name (default: gpt-5.4).",
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
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        metavar="N",
        help="Max concurrent file evaluations in folder mode (default: 5).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cache" / "vision_capability",
        help=(
            "Store per-step verdict JSON here to skip repeat judge API calls for "
            "unchanged steps (default: intrinsic_metric/cache/vision_capability/)."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    client = _make_client()
    cache = load_cache(args.cache_dir)

    # ---- inspect mode (single file) ----
    if args.inspect is not None:
        html_path = args.inspect.expanduser().resolve()
        if not html_path.is_file():
            print(f"Error: not a file: {html_path}", file=sys.stderr)
            return 1

        config, verdicts = evaluate_file(
            html_path, client, args.model, args.cache_dir, cache
        )
        traj = aggregate_verdicts(html_path, verdicts)
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        json_data = {
            "generated": generated,
            "file": str(html_path),
            "intent": config.intent,
            "task_image": config.image_path,
            "total_steps": traj.total_steps,
            "vision_good": traj.vision_good,
            "vision_bad": traj.vision_bad,
            "vision_na": traj.vision_na,
            "steps": [asdict(v) for v in traj.step_verdicts],
        }

        print(json.dumps(json_data, ensure_ascii=False, indent=2))

        if not args.no_save:
            out_path = args.output or (
                Path(__file__).resolve().parent
                / "results"
                / f"{METRIC_NAME}_{html_path.parent.name}_{html_path.stem}.json"
            )
            out_path = out_path.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(json_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote inspect JSON to: {out_path}", file=sys.stderr)

        return 0

    # ---- folder mode ----
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

    aclient = _make_async_client()
    semaphore = asyncio.Semaphore(args.concurrent)

    async def _run_all() -> list[tuple[Path, TaskConfig, list[StepVerdict]]]:
        tasks = [
            _eval_one(f, aclient, args.model, args.cache_dir, cache, semaphore)
            for f in html_files
        ]
        return await asyncio.gather(*tasks)

    async def _eval_one(
        f: Path,
        ac: AsyncOpenAI,
        model: str,
        cd: Path,
        c: dict[str, dict[str, Any]],
        sem: asyncio.Semaphore,
    ) -> tuple[Path, TaskConfig, list[StepVerdict]]:
        cfg, verdicts = await async_evaluate_file(f, ac, model, cd, c, sem)
        return f, cfg, verdicts

    raw_results = asyncio.run(_run_all())

    results: list[TrajectoryResult] = []
    for html_f, _cfg, verdicts in raw_results:
        traj = aggregate_verdicts(html_f, verdicts)
        results.append(traj)

    folder_result = aggregate_folder(results)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    json_data = {
        "generated": generated,
        "folder": str(folder),
        "total_steps": folder_result.total_steps,
        "vision_good": folder_result.vision_good,
        "vision_bad": folder_result.vision_bad,
        "vision_na": folder_result.vision_na,
        "num_files": folder_result.num_files,
        "per_file": [
            {
                "file": str(r.file_path),
                "total_steps": r.total_steps,
                "vision_good": r.vision_good,
                "vision_bad": r.vision_bad,
                "vision_na": r.vision_na,
            }
            for r in folder_result.per_file
        ],
    }

    print(json.dumps(json_data, ensure_ascii=False, indent=2))

    if not args.no_save:
        out_path = args.output or (
            Path(__file__).resolve().parent
            / "results"
            / f"{METRIC_NAME}_{folder.name}.json"
        )
        out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote JSON summary to: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
