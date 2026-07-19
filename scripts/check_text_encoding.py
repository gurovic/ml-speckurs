#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail on likely encoding corruption in text files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUSPICIOUS_PATTERNS = (
    "????",
    "\ufffd",  # replacement character
    "Ð",
    "Ñ",
    "Ã",
)

CHECK_EXTENSIONS = {".ipynb", ".md", ".py", ".mdc", ".json", ".yaml", ".yml", ".txt"}
SKIP_DIRS = {
    ".git",
    ".codex_competition_pydeps",
    ".codex_mpl_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "catboost_info",
}
IGNORE_LINE_CONTAINS = (
    "SUSPICIOUS_PATTERNS",
    "кракозябр",
    "suspicious encoding pattern",
)


def staged_files(repo_root: Path) -> list[Path]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    out = subprocess.check_output(cmd, cwd=repo_root, text=True, encoding="utf-8")
    return [repo_root / line.strip() for line in out.splitlines() if line.strip()]


def all_files(repo_root: Path) -> list[Path]:
    return [p for p in repo_root.rglob("*") if p.is_file()]


def should_check(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    return path.suffix.lower() in CHECK_EXTENSIONS


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        errors.append(f"{path}: not valid UTF-8 ({e})")
        return errors
    except OSError as e:
        errors.append(f"{path}: read error ({e})")
        return errors

    inside_pattern_block = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        if "SUSPICIOUS_PATTERNS" in line and "=" in line:
            inside_pattern_block = True
            continue
        if inside_pattern_block and line.strip() == ")":
            inside_pattern_block = False
            continue
        if inside_pattern_block:
            continue
        if any(marker in line for marker in IGNORE_LINE_CONTAINS):
            continue
        for pat in SUSPICIOUS_PATTERNS:
            if pat in line:
                errors.append(
                    f"{path}:{line_no}: suspicious encoding pattern `{pat}`"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="check only staged files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    files = staged_files(repo_root) if args.staged else all_files(repo_root)
    files = [p for p in files if should_check(p)]

    all_errors: list[str] = []
    for path in files:
        all_errors.extend(check_file(path))

    if all_errors:
        print("Encoding check failed:\n", file=sys.stderr)
        for err in all_errors:
            print(f"- {err}", file=sys.stderr)
        print(
            "\nFix file encoding (UTF-8) and remove mojibake before commit.",
            file=sys.stderr,
        )
        return 1

    print(f"Encoding check passed ({len(files)} files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

