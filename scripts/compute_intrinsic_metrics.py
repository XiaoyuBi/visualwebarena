import re
import sys
from collections import Counter
from pathlib import Path


def compute_url_revisit_rate(folder: str) -> tuple[float, int]:
    """Compute URL revisit rate from a folder of render HTML files.

    For each HTML file, extracts the trajectory (sequence of URLs visited),
    counts excess revisits (total visits minus 1 for each URL visited more
    than once), and divides by trajectory length.

    Args:
        folder: Path to a folder containing render HTML files.

    Returns:
        A tuple of (sum_of_ratios, num_htmls) where:
          - sum_of_ratios: sum over all files of (num_excess_revisits / trajectory_length)
          - num_htmls: total number of HTML files in the folder
    """
    url_pattern = re.compile(r"<h3 class='url'>.*?URL:\s*(.*?)</a></h3>")

    html_files = sorted(Path(folder).glob("*.html"))
    num_htmls = len(html_files)
    sum_of_ratios = 0.0

    for html_file in html_files:
        content = html_file.read_text()
        urls = url_pattern.findall(content)
        traj_length = len(urls)
        if traj_length == 0:
            continue
        counts = Counter(urls)
        num_excess_revisits = sum(c - 1 for c in counts.values() if c > 1)
        sum_of_ratios += num_excess_revisits / traj_length

    return sum_of_ratios, num_htmls


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_intrinsic_metrics.py <folder>")
        sys.exit(1)

    folder = sys.argv[1]
    ratio_sum, count = compute_url_revisit_rate(folder)
    print(f"Sum of (url_revisit_rate): {ratio_sum:.4f}")
    print(f"Number of HTML files:      {count}")
    print(f"Average url_revisit_rate:  {ratio_sum / count:.4f}" if count > 0 else "No HTML files found.")
