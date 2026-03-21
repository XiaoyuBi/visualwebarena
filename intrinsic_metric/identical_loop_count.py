"""Extract loop segments from VisualWebArena render HTML and count duplicate patterns.

Consecutive identical URLs are collapsed before loop detection so ``AAA`` is not
treated as a navigational loop.

New Page URLs are written by ``RenderHelper`` in ``browser_env/helper_functions.py`` as::

    <h3 class='url'><a href={url}>URL: {url}</a></h3>

with an unquoted ``href`` value (no ``>`` inside the URL).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# CLI default output filenames and report header use this label.
METRIC_NAME = "identical_loop_count"

# Matches each New Page URL line produced by RenderHelper.render()
_NEW_PAGE_URL_RE = re.compile(
    r"<h3 class='url'><a href=([^>]+)>",
    re.IGNORECASE,
)


def extract_new_page_urls_from_render_html(html: str) -> list[str]:
    """Return ordered list of page URLs, one per ``New Page`` block."""
    return _NEW_PAGE_URL_RE.findall(html)


def dedupe_consecutive_urls(urls: list[str]) -> list[str]:
    """Collapse runs of identical consecutive URLs to a single entry (e.g. AAA -> A).

    Render logs often repeat the same URL across steps when the page did not
    meaningfully change; those should not form a "loop" like A-B-A.
    """
    if not urls:
        return []
    out: list[str] = [urls[0]]
    for u in urls[1:]:
        if u != out[-1]:
            out.append(u)
    return out


def extract_loops_from_urls(urls: list[str]) -> list[tuple[str, ...]]:
    """Greedy non-overlapping loop extraction on a *consecutive-deduped* URL list.

    First applies :func:`dedupe_consecutive_urls` so repeated observations of the
    same URL in a row (``AA``, ``AAA``, …) are not treated as multi-step loops.

    A *loop* is a contiguous segment ``urls[s:e+1]`` with ``urls[s] == urls[e]``
    and ``e - s >= 2`` (at least three *deduped* steps: start, at least one other
    URL, return). Pure consecutive repeats are never loops.

    Algorithm: from current index ``s``, take the *smallest* ``e`` such that
    ``e >= s + 2`` and ``urls[e] == urls[s]``; emit that segment and set
    ``s = e + 1``. If none exists, ``s += 1``.
    """
    urls = dedupe_consecutive_urls(urls)
    if not urls:
        return []

    loops: list[tuple[str, ...]] = []
    s = 0
    n = len(urls)

    while s < n:
        e = None
        for j in range(s + 2, n):
            if urls[j] == urls[s]:
                e = j
                break
        if e is not None:
            loops.append(tuple(urls[s : e + 1]))
            s = e + 1
        else:
            s += 1

    return loops


def count_identical_loops(loops: list[tuple[str, ...]]) -> int:
    """Sum of occurrence counts for each loop pattern that appears at least twice.

    Example: ``ABA`` x3 and ``ABCA`` x1 -> ``3``; ``ABA`` x3 and ``ABCA`` x2 -> ``5``.
    """
    if not loops:
        return 0
    counts = Counter(loops)
    return sum(c for c in counts.values() if c >= 2)


def per_file_identical_loop_counts(folder: str | Path) -> list[tuple[Path, int]]:
    """For each ``render_*.html`` under ``folder``, compute that file's identical-loop count.

    Returns:
        Sorted list of ``(absolute_path, identical_loop_count)``. Non-directory -> ``[]``.
    """
    path = Path(folder)
    if not path.is_dir():
        return []

    rows: list[tuple[Path, int]] = []
    for f in sorted(path.glob("render_*.html")):
        html = f.read_text(encoding="utf-8", errors="replace")
        urls = extract_new_page_urls_from_render_html(html)
        loops = extract_loops_from_urls(urls)
        loop_count = count_identical_loops(loops)
        rows.append((f.resolve(), loop_count))
    return rows


def aggregate_identical_loops_from_render_folder(folder: str | Path) -> tuple[int, int]:
    """Sum duplicate-loop counts over all ``render_*.html`` files in ``folder``.

    Returns:
        ``(total_identical_loop_count, num_render_html_files)``.
        Empty folder -> ``(0, 0)``.
    """
    rows = per_file_identical_loop_counts(folder)
    return (sum(s for _, s in rows), len(rows))


def format_inspect_report(html_path: Path, urls: list[str], loops: list[tuple[str, ...]]) -> str:
    """Human-readable listing of extracted New Page URLs and greedy loop segments."""
    urls_deduped = dedupe_consecutive_urls(urls)
    lines: list[str] = [
        f"metric: {METRIC_NAME}",
        f"file: {html_path.resolve()}",
        f"new_page_url_count (raw): {len(urls)}",
        f"url_count_after_consecutive_dedupe: {len(urls_deduped)}",
        "",
        "## URLs — raw from HTML (order = trajectory order)",
    ]
    for i, u in enumerate(urls):
        lines.append(f"  [{i}] {u}")

    lines.extend(
        [
            "",
            "## URLs — after consecutive dedupe (used for loop detection)",
        ]
    )
    for i, u in enumerate(urls_deduped):
        lines.append(f"  [{i}] {u}")

    lines.extend(
        [
            "",
            "## Loops (on deduped sequence; greedy; start URL == end URL; len >= 3)",
            f"loop_segment_count: {len(loops)}",
        ]
    )
    for i, loop in enumerate(loops):
        lines.append(f"  loop {i} (len={len(loop)}):")
        for j, u in enumerate(loop):
            lines.append(f"    [{j}] {u}")

    counts = Counter(loops)
    identical_count = count_identical_loops(loops)
    lines.extend(
        [
            "",
            "## Duplicate-pattern count (this file)",
            f"identical_loop_count: {identical_count}",
            "## Loop pattern counts",
        ]
    )
    for pattern, c in counts.most_common():
        # shorten display: show first/last url + length if very long
        if len(pattern) <= 5:
            lines.append(f"  count={c}  pattern={pattern!r}")
        else:
            lines.append(
                f"  count={c}  len={len(pattern)}  "
                f"start={pattern[0]!r} ... end={pattern[-1]!r}"
            )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI: aggregate identical-loop count over ``render_*.html`` in a results folder."""
    parser = argparse.ArgumentParser(
        description=(
            "Count duplicate loop patterns from New Page URLs in render_*.html files "
            "(VisualWebArena RenderHelper output)."
        )
    )
    parser.add_argument(
        "--inspect",
        type=Path,
        metavar="HTML",
        default=None,
        help=(
            "Single render_*.html file: print extracted New Page URLs and loop segments, "
            "then exit (ignores positional folder)."
        ),
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing render_*.html files (e.g. a single run results folder).",
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

    if args.inspect is not None:
        html_path = args.inspect.expanduser().resolve()
        if not html_path.is_file():
            print(f"Error: not a file: {html_path}", file=sys.stderr)
            return 1
        html = html_path.read_text(encoding="utf-8", errors="replace")
        urls = extract_new_page_urls_from_render_html(html)
        loops = extract_loops_from_urls(urls)
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = format_inspect_report(html_path, urls, loops)
        report = f"generated: {generated}\n---\n{body}"

        print(report, end="")

        if not args.no_save:
            out_path = args.output
            if out_path is None:
                out_path = (
                    Path(__file__).resolve().parent
                    / "results"
                    / f"{METRIC_NAME}_inspect_{html_path.stem}.txt"
                )
            else:
                out_path = out_path.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report, encoding="utf-8")
            print(f"Wrote inspect report to: {out_path}", file=sys.stderr)
        return 0

    if args.folder is None:
        parser.error("the following arguments are required: folder (unless using --inspect)")

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        return 1

    rows = per_file_identical_loop_counts(folder)
    total_count = sum(s for _, s in rows)
    n_files = len(rows)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"metric: {METRIC_NAME}",
        f"folder: {folder}",
        f"generated: {generated}",
        "---",
        f"total_identical_loop_count: {total_count}",
        f"render_html_files: {n_files}",
        "",
        "## html_paths_with_identical_loops (per-file count > 0)",
    ]
    hits = [(p, s) for p, s in rows if s > 0]
    if not hits:
        lines.append("  (none)")
    else:
        for p, s in hits:
            lines.append(f"  {p}  identical_loop_count: {s}")
    lines.append("")
    report = "\n".join(lines) + "\n"

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
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
