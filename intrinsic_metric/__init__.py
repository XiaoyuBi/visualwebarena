"""Intrinsic trajectory metrics (e.g. from render HTML)."""

from .identical_loop_count import (
    aggregate_identical_loops_from_render_folder,
    count_identical_loops,
    dedupe_consecutive_urls,
    extract_loops_from_urls,
    extract_new_page_urls_from_render_html,
    per_file_identical_loop_counts,
)

__all__ = [
    "aggregate_identical_loops_from_render_folder",
    "count_identical_loops",
    "dedupe_consecutive_urls",
    "extract_loops_from_urls",
    "extract_new_page_urls_from_render_html",
    "per_file_identical_loop_counts",
]
