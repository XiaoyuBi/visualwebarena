"""Per-file URL revisit rate from VisualWebArena ``render_*.html`` trajectories."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

METRIC_NAME = "url_revisit_rate"

# Visible URL text after "URL: " in RenderHelper output (same line as href).
_URL_LINE_RE = re.compile(
    r"<h3 class='url'>.*?URL:\s*(.*?)</a></h3>",
    re.DOTALL | re.IGNORECASE,
)


def compute_url_revisit_rate(folder: str | Path) -> tuple[float, int]:
    """Sum over ``render_*.html`` of (excess revisits / trajectory length).

    Excess revisits = for each URL seen more than once, add (count - 1).
    Trajectory length = number of New Page URL observations in the file.
    """
    path = Path(folder)
    if not path.is_dir():
        return (0.0, 0)

    html_files = sorted(path.glob("render_*.html"))
    num_htmls = len(html_files)
    sum_of_ratios = 0.0

    for html_file in html_files:
        content = html_file.read_text(encoding="utf-8", errors="replace")
        urls = _URL_LINE_RE.findall(content)
        traj_length = len(urls)
        if traj_length == 0:
            continue
        counts = Counter(urls)
        num_excess_revisits = sum(c - 1 for c in counts.values() if c > 1)
        sum_of_ratios += num_excess_revisits / traj_length

    return sum_of_ratios, num_htmls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute URL revisit rate from render_*.html in a results folder "
            "(excess revisits / trajectory length per file, then sum)."
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
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        return 1

    ratio_sum, count = compute_url_revisit_rate(folder)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"metric: {METRIC_NAME}",
        f"folder: {folder}",
        f"generated: {generated}",
        "---",
        f"sum_url_revisit_rate: {ratio_sum:.6f}",
        f"render_html_files: {count}",
    ]
    if count > 0:
        lines.append(f"average_url_revisit_rate: {ratio_sum / count:.6f}")
    else:
        lines.append("average_url_revisit_rate: (no render_*.html files)")
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
