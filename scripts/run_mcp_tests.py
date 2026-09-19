"""Run MCP-only contracts, not historical training/asset acceptance."""

from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = sorted((ROOT / "tests").glob("test_mcp*.py"))
    paths += [
        ROOT / "tests" / name
        for name in [
            "test_figure_candidate_preannotator.py",
            "test_figure_candidate_review.py",
            "test_visual_batch_review.py",
        ]
        if (ROOT / "tests" / name).is_file()
    ]
    return subprocess.call(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "pytest",
            *(str(p) for p in paths),
            "-q",
            "-p",
            "no:cacheprovider",
            *sys.argv[1:],
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


if __name__ == "__main__":
    raise SystemExit(main())
