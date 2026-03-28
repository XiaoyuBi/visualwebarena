"""Per-file *refined* invalid action rate from VisualWebArena ``render_*.html`` trajectories.

Unlike the basic ``invalid_action_rate`` which counts **all** transitions with
identical consecutive URLs, this refined version:

1. Parses the action at each step (canonical ``scroll [down]`` format **and**
   Playwright ``page.…().click()`` format).
2. Skips actions that are inherently non-URL-changing (``scroll``, ``none``,
   ``stop``, ``type``).
3. Only counts a transition as *invalid* when a URL-changing action (``click``,
   ``goto``, ``go_back``, ``press``, ``new_tab``, …) fails to change the URL.

The rate for a single file is::

    invalid_count / url_changing_transitions

Files with no URL-changing transitions are skipped (rate 0).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from intrinsic_metric.vision_capability import parse_render_html

METRIC_NAME = "invalid_action_rate_refined"

_NON_URL_CHANGING = frozenset({"scroll", "none", "stop", "type"})


def _extract_action_type(action_str: str) -> str:
    """Extract a high-level action type from a canonical or Playwright-style string."""
    action = action_str.strip().lower()
    if not action:
        return "unknown"

    first_word = action.split()[0]
    _CANONICAL = {
        "scroll", "click", "goto", "go_back", "type", "press",
        "none", "stop", "new_tab", "close_tab", "tab_focus", "page_focus",
    }
    if first_word in _CANONICAL:
        return first_word

    if ".click(" in action:
        return "click"
    if ".fill(" in action or ".type(" in action:
        return "type"
    if ".scroll" in action:
        return "scroll"
    if ".press(" in action:
        return "press"
    if "goto(" in action:
        return "goto"
    if "go_back" in action:
        return "go_back"

    return "unknown"


def compute_refined_invalid_action_rate(
    folder: str | Path,
) -> tuple[float, int, list[dict]]:
    """Sum per-file refined invalid-action rates over ``render_*.html`` in *folder*.

    Returns ``(sum_of_rates, num_render_html_files, per_file_details)``.
    """
    path = Path(folder)
    if not path.is_dir():
        return 0.0, 0, []

    html_files = sorted(path.glob("render_*.html"))
    num_htmls = len(html_files)
    sum_of_rates = 0.0
    per_file: list[dict] = []

    for html_file in html_files:
        content = html_file.read_text(encoding="utf-8", errors="replace")
        _config, steps = parse_render_html(content)

        if len(steps) < 2:
            per_file.append({
                "file": html_file.name,
                "total_transitions": 0,
                "url_changing_transitions": 0,
                "invalid_count": 0,
                "rate": 0.0,
            })
            continue

        url_changing_transitions = 0
        invalid_count = 0

        for i in range(len(steps) - 1):
            action_str = steps[i].parsed_action
            if not action_str.strip():
                action_str = steps[i + 1].prev_action

            action_type = _extract_action_type(action_str)
            if action_type in _NON_URL_CHANGING:
                continue

            url_changing_transitions += 1
            if steps[i].url == steps[i + 1].url:
                invalid_count += 1

        if url_changing_transitions > 0:
            rate = invalid_count / url_changing_transitions
        else:
            rate = 0.0

        sum_of_rates += rate
        per_file.append({
            "file": html_file.name,
            "total_transitions": len(steps) - 1,
            "url_changing_transitions": url_changing_transitions,
            "invalid_count": invalid_count,
            "rate": rate,
        })

    return sum_of_rates, num_htmls, per_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute refined invalid action rate from render_*.html in a results "
            "folder. Excludes non-URL-changing actions (scroll, none, stop, type) "
            "and only counts as invalid when a URL-changing action fails to "
            "change the URL."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory containing render_*.html files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Write report to this file. Default: intrinsic_metric/results/"
            f"{METRIC_NAME}_<folder_name>.txt next to this module."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a report file (only print to stdout).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file breakdown.",
    )
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        return 1

    ratio_sum, count, per_file = compute_refined_invalid_action_rate(folder)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"metric: {METRIC_NAME}",
        f"folder: {folder}",
        f"generated: {generated}",
        f"non_url_changing_actions_excluded: {sorted(_NON_URL_CHANGING)}",
        "---",
        f"sum_invalid_action_rate: {ratio_sum:.6f}",
        f"render_html_files: {count}",
    ]
    if count > 0:
        lines.append(f"average_invalid_action_rate: {ratio_sum / count:.6f}")
    else:
        lines.append("average_invalid_action_rate: (no render_*.html files)")

    if args.verbose and per_file:
        lines.append("")
        lines.append("--- per-file breakdown ---")
        for entry in per_file:
            lines.append(
                f"  {entry['file']}: "
                f"total_trans={entry['total_transitions']}  "
                f"url_changing={entry['url_changing_transitions']}  "
                f"invalid={entry['invalid_count']}  "
                f"rate={entry['rate']:.4f}"
            )

    lines.append("")
    report = "\n".join(lines)

    print(report, end="")

    if not args.no_save:
        out_path = args.output
        if out_path is None:
            out_path = (
                Path(__file__).resolve().parent
                / "results"
                / f"{METRIC_NAME}_{folder.name}.txt"
            )
        else:
            out_path = out_path.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote report to: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
