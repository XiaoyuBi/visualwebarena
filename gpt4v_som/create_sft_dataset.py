#!/usr/bin/env python3
"""Build Fireworks AI VLM SFT JSONL from VisualWebArena render_*.html trajectories.

Each line is one agent step (single-turn, OpenAI-compatible chat format with
optional image_url parts), suitable for fine-tuning per:
https://docs.fireworks.ai/fine-tuning/fine-tuning-vlm#multi-turn-conversation

One HTML file yields N lines (one per usable step), matching the stateless
single-turn inference pattern of the agent: each LLM call sees only the current
observation, not prior steps.

User/assistant layout for SoM matches ``MultimodalCoTPromptConstructor`` in
``agent/prompts/prompt_constructor.py`` (OpenAI chat): after the template text,
``IMAGES: (1) current page screenshot`` then the page screenshot ``image_url``.

Usage (from repo root)::

    python gpt4v_som/create_sft_dataset.py \\
        --raw-dir gpt4v_som/raw \\
        -o gpt4v_som/dataset/sft_trajectories.jsonl

    ``--min-render-id`` defaults to 100; use ``0`` for no id filter.

    # Compact images: real ``image_url`` with a tiny valid PNG + content id line::

    python gpt4v_som/create_sft_dataset.py --ignore-images --min-render-id 400 -o out.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Same strings as MultimodalCoTPromptConstructor.get_lm_api_input (OpenAI provider).
_CURRENT_PAGE_IMAGE_LABEL = "IMAGES: (1) current page screenshot"
_EXAMPLE_SCREENSHOT_OMITTED = "(example screenshot omitted)"

# ---------------------------------------------------------------------------
# HTML parsing (aligned with intrinsic_metric/vision_capability.py; local copy
# so this script does not import vision_capability / openai).
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
_RENDER_FILENAME_RE = re.compile(r"^render_(\d+)\.html$", re.IGNORECASE)
_RESULT_RE = re.compile(r"\[Result\]\s+\((PASS|FAIL)\)\s+\S+/(\d+)\.json")


def render_id_from_filename(path: Path) -> int | None:
    """Numeric suffix of ``render_<id>.html``, or None if the name does not match."""
    m = _RENDER_FILENAME_RE.match(path.name)
    return int(m.group(1)) if m else None


def _load_results(subdir: Path) -> dict[int, bool]:
    """Parse ``results.txt`` in *subdir* into ``{task_id: True/False}``."""
    txt = subdir / "results.txt"
    if not txt.is_file():
        return {}
    out: dict[int, bool] = {}
    for line in txt.read_text(encoding="utf-8").splitlines():
        m = _RESULT_RE.search(line)
        if m:
            out[int(m.group(2))] = (m.group(1) == "PASS")
    return out


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


def _parse_config(config_text: str) -> TaskConfig:
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
    parts = _NEW_PAGE_RE.split(html)
    if len(parts) < 2:
        config_match = _CONFIG_PRE_RE.search(parts[0] if parts else "")
        config = _parse_config(config_match.group(1) if config_match else "")
        return config, []

    preamble = parts[0]
    config_match = _CONFIG_PRE_RE.search(preamble)
    config = _parse_config(config_match.group(1) if config_match else "")

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

DEFAULT_PROMPT = (
    _REPO_ROOT / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"
)
DEFAULT_OUT = _REPO_ROOT / "gpt4v_som/dataset/sft_trajectories.jsonl"
DEFAULT_RAW = _REPO_ROOT / "gpt4v_som/raw"


def _truncate(s: str, max_obs_chars: int) -> str:
    if len(s) <= max_obs_chars:
        return s
    return s[:max_obs_chars]


# Valid 1×1 PNG (~70 bytes raw); keeps OpenAI-style ``image_url`` without large payloads.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAKmSdWQAAAABJRU5ErkJggg=="
)
TINY_PLACEHOLDER_IMAGE_URL = f"data:image/png;base64,{_TINY_PNG_B64}"


def _screenshot_content_id(b64: str) -> str:
    """Short stable id from decoded image bytes (distinct per real screenshot)."""
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        raw = b64.encode("ascii", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:12]


def _compact_image_note(b64: str) -> str:
    """Text line paired with the tiny PNG so rows stay identifiable."""
    cid = _screenshot_content_id(b64)
    return (
        "SCREENSHOT_CONTENT_ID: sha256:"
        + cid
        + " (full-resolution pixels omitted for dataset size; preceding image_url is a 1x1 PNG placeholder.)"
    )


def _resize_b64_image(b64: str, max_side: int) -> str:
    """Resize a base64-encoded image so its longest side <= max_side pixels.

    Returns the original b64 unchanged if the image is already within bounds,
    if Pillow is unavailable, or if decoding fails.
    """
    try:
        import io
        from PIL import Image
    except ImportError:
        return b64
    try:
        raw = base64.b64decode(b64, validate=False)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if max(w, h) <= max_side:
            return b64
        scale = max_side / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return b64


def _tiny_placeholder_image_part() -> dict[str, Any]:
    """Same multimodal shape as real VLM data: ``type: image_url`` with a minimal valid PNG."""
    return {
        "type": "image_url",
        "image_url": {"url": TINY_PLACEHOLDER_IMAGE_URL},
    }


def _load_image_data_url(repo_root: Path, rel_path: str) -> str | None:
    """Load image file as data:image/png;base64,... URL."""
    p = Path(rel_path)
    if not p.is_absolute():
        p = repo_root / p
    if not p.is_file():
        return None
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suf = p.suffix.lower()
    mime = "image/png"
    if suf in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suf == ".webp":
        mime = "image/webp"
    elif suf == ".gif":
        mime = "image/gif"
    return f"data:{mime};base64,{b64}"


def _user_message_from_template(
    template: str,
    observation: str,
    url: str,
    objective: str,
    previous_action: str,
    screenshot_b64: str | None,
    *,
    ignore_images: bool = False,
    resize_max: int | None = None,
) -> dict[str, Any]:
    text = template.format(
        observation=observation,
        url=url,
        objective=objective,
        previous_action=previous_action,
    )
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if screenshot_b64:
        parts.append({"type": "text", "text": _CURRENT_PAGE_IMAGE_LABEL})
        if ignore_images:
            parts.append(_tiny_placeholder_image_part())
        else:
            img_b64 = (
                _resize_b64_image(screenshot_b64, resize_max)
                if resize_max is not None
                else screenshot_b64
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                    },
                }
            )
    return {"role": "user", "content": parts}


def _few_shot_messages(
    examples: list,
    repo_root: Path,
    *,
    ignore_images: bool = False,
    resize_max: int | None = None,
) -> list[dict[str, Any]]:
    """Turn prompt JSON `examples` into user/assistant message pairs."""
    out: list[dict[str, Any]] = []
    for ex in examples:
        if len(ex) < 3:
            continue
        user_text, assistant_text, image_rel = ex[0], ex[1], ex[2]
        user_text = html.unescape(str(user_text))
        assistant_text = html.unescape(str(assistant_text))
        data_url = _load_image_data_url(repo_root, str(image_rel))
        if data_url:
            if ignore_images:
                few_user_parts: list[dict[str, Any]] = [
                    {"type": "text", "text": user_text},
                    {"type": "text", "text": _CURRENT_PAGE_IMAGE_LABEL},
                    _tiny_placeholder_image_part(),
                ]
            else:
                if resize_max is not None:
                    # data_url is "data:<mime>;base64,<b64>" — resize the b64 part
                    _prefix, _b64 = data_url.split(",", 1)
                    _b64_resized = _resize_b64_image(_b64, resize_max)
                    data_url = f"data:image/png;base64,{_b64_resized}"
                few_user_parts = [
                    {"type": "text", "text": user_text},
                    {"type": "text", "text": _CURRENT_PAGE_IMAGE_LABEL},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ]
            out.append(
                {
                    "role": "user",
                    "content": few_user_parts,
                }
            )
        else:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "text", "text": _EXAMPLE_SCREENSHOT_OMITTED},
                    ],
                }
            )
        out.append(
            {"role": "assistant", "content": assistant_text, "weight": 0},
        )
    return out


def build_step_examples(
    config: TaskConfig,
    steps: list[StepData],
    prompt_data: dict[str, Any],
    *,
    include_examples: bool,
    max_obs_chars: int,
    repo_root: Path,
    ignore_images: bool = False,
    resize_max: int | None = None,
) -> list[dict[str, Any]]:
    """One JSON object per usable step: { \"messages\": [...] }.

    Each object is a single-turn example matching the stateless agent: the LLM
    sees [system intro] + [few-shot turns] + [one user message] and must predict
    [one assistant message].  The caller is responsible for adding ``success``
    and ``meta`` fields to each returned object.
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

    # Build the static prefix once (system + few-shot); reused for every step.
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

    out: list[dict[str, Any]] = []
    for step in steps:
        raw = step.raw_prediction.strip()
        if not raw:
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
        messages: list[dict[str, Any]] = list(prefix) + [
            user_msg,
            {"role": "assistant", "content": html.unescape(raw)},
        ]
        out.append({"messages": messages, "_step_index": step.step_index})

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert render_*.html trajectories to Fireworks VLM SFT JSONL.",
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
        help="Output JSONL path (default: gpt4v_som/dataset/sft_trajectories.jsonl)",
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
        help="Max characters for accessibility-tree text per step (default: 15360 ≈ 3840 tokens × 4 chars/token)",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Omit few-shot example turns from each trajectory",
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
            "Use 0 to disable and include every render_*.html. "
            "When filtering, files that do not match the pattern are excluded."
        ),
    )
    parser.add_argument(
        "--max-render-id",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only include render_<id>.html with numeric id < N (exclusive upper bound). "
            "Use together with --min-render-id 0 to select an eval slice, e.g. idx < 100."
        ),
    )
    parser.add_argument(
        "--ignore-images",
        action="store_true",
        help=(
            "Replace every screenshot with a minimal valid 1x1 PNG image_url "
            "instead of the full screenshot (small on disk, same JSON shape as VLM SFT)."
        ),
    )
    parser.add_argument(
        "--image-resize",
        type=int,
        default=None,
        metavar="MAX_SIDE",
        help=(
            "Resize screenshots (and few-shot example images) so the longest side "
            "is at most MAX_SIDE pixels before base64-encoding. "
            "Aspect ratio is preserved. Requires Pillow. "
            "Ignored when --ignore-images is set. Example: --image-resize 720"
        ),
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help=(
            "Only emit trajectories where results.txt records PASS. "
            "Trajectories with no results.txt entry are also skipped."
        ),
    )
    parser.add_argument(
        "--with-meta",
        action="store_true",
        help="Add meta (source path, task_id, domain) to each JSON line",
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

    # Collect HTML files grouped by subdirectory so we can load results.txt once per domain.
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

    # Pre-load results.txt for every subdirectory that contains HTML files we plan to process.
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
            # None means no entry in results.txt; True/False means PASS/FAIL.
            success: bool | None = sub_results.get(rid) if rid is not None else None

            if args.success_only and success is not True:
                skipped += 1
                continue

            try:
                text = html_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                print(f"Warning: skip {html_path}: {e}", file=sys.stderr)
                skipped += 1
                continue

            config, steps = parse_render_html(text)
            resize_max = args.image_resize if not args.ignore_images else None
            step_examples = build_step_examples(
                config,
                steps,
                prompt_data,
                include_examples=not args.no_examples,
                max_obs_chars=args.max_obs_chars,
                repo_root=_REPO_ROOT,
                ignore_images=args.ignore_images,
                resize_max=resize_max,
            )
            if not step_examples:
                skipped += 1
                continue

            for example in step_examples:
                step_index = example.pop("_step_index")
                line_obj: dict[str, Any] = dict(example)
                line_obj["success"] = success
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
