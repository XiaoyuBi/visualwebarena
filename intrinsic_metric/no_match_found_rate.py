"""Per-file "No match found" rate from VisualWebArena ``render_*.html`` trajectories.

A step has "No match found" when the agent's action references an element_id
that does not exist in the current observation's node info.  This text appears
inside the ``<div class='parsed_action'>`` block rendered by the harness.

The rate for a single file is::

    no_match_count / total_action_steps

Files with zero action steps are skipped.

Usage::

    python intrinsic_metric/no_match_found_rate.py results/shopping/shopping_gpt5mini_som_0_100/
    python intrinsic_metric/no_match_found_rate.py --inspect results/shopping/shopping_gpt5mini_som_0_100/render_11.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRIC_NAME = "no_match_found_rate"

_PARSED_ACTION_RE = re.compile(
    r"<div class='parsed_action'[^>]*>\s*<pre>(.*?)</pre>\s*</div>",
    re.DOTALL | re.IGNORECASE,
)

_NO_MATCH_FOUND = "No match found"


@dataclass
class FileResult:
    file: str
    task_id: str
    total_steps: int
    no_match_count: int
    no_match_rate: float
    no_match_step_indices: list[int]


def _extract_task_id(html: str) -> str:
    m = re.search(r"task_id:\s*(\S+)", html)
    return m.group(1) if m else "unknown"


def _split_step_blocks(html: str) -> list[str]:
    """Split on ``<h2>New Page</h2>`` and return one fragment per step."""
    parts = re.split(r"<h2>New Page</h2>", html, flags=re.IGNORECASE)
    return parts[1:] if len(parts) > 1 else []


def evaluate_render_html(html: str, html_path: Path) -> FileResult:
    task_id = _extract_task_id(html)
    step_blocks = _split_step_blocks(html)

    no_match_count = 0
    no_match_indices: list[int] = []

    for i, block in enumerate(step_blocks):
        m = _PARSED_ACTION_RE.search(block)
        if m and _NO_MATCH_FOUND in m.group(1):
            no_match_count += 1
            no_match_indices.append(i)

    total = len(step_blocks)
    rate = no_match_count / total if total > 0 else 0.0

    return FileResult(
        file=str(html_path),
        task_id=task_id,
        total_steps=total,
        no_match_count=no_match_count,
        no_match_rate=rate,
        no_match_step_indices=no_match_indices,
    )


def aggregate_folder(results: list[FileResult]) -> dict[str, Any]:
    n = len(results)
    total_steps = sum(r.total_steps for r in results)
    total_no_match = sum(r.no_match_count for r in results)
    sum_of_rates = sum(r.no_match_rate for r in results)
    files_with_no_match = sum(1 for r in results if r.no_match_count > 0)

    return {
        "num_files": n,
        "total_steps": total_steps,
        "total_no_match_found": total_no_match,
        "overall_no_match_rate": total_no_match / total_steps if total_steps else None,
        "sum_per_file_no_match_rate": sum_of_rates,
        "avg_per_file_no_match_rate": sum_of_rates / n if n else None,
        "files_with_no_match": files_with_no_match,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Count "No match found" occurrences in render_*.html trajectories '
            "(agent referenced an element_id absent from the observation)."
        )
    )
    parser.add_argument(
        "--inspect",
        type=Path,
        metavar="HTML",
        default=None,
        help="Single render_*.html: print detailed JSON.",
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

    # ---- inspect mode ----
    if args.inspect is not None:
        html_path = args.inspect.expanduser().resolve()
        if not html_path.is_file():
            print(f"Error: not a file: {html_path}", file=sys.stderr)
            return 1
        html = html_path.read_text(encoding="utf-8", errors="replace")
        result = evaluate_render_html(html, html_path)

        payload: dict[str, Any] = {
            "generated": generated,
            "file": result.file,
            "task_id": result.task_id,
            "total_steps": result.total_steps,
            "no_match_count": result.no_match_count,
            "no_match_rate": result.no_match_rate,
            "no_match_step_indices": result.no_match_step_indices,
        }

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

    results: list[FileResult] = []
    per_file: list[dict[str, Any]] = []

    for f in html_files:
        html = f.read_text(encoding="utf-8", errors="replace")
        r = evaluate_render_html(html, f)
        results.append(r)
        per_file.append({
            "file": r.file,
            "task_id": r.task_id,
            "total_steps": r.total_steps,
            "no_match_count": r.no_match_count,
            "no_match_rate": r.no_match_rate,
            "no_match_step_indices": r.no_match_step_indices,
        })

    agg = aggregate_folder(results)

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
