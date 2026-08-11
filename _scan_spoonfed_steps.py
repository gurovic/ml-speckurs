# -*- coding: utf-8 -*-
"""Find assignment steps that look like spoon-fed solution code."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"c:\Users\Гуровиц Владимир\ml-speckurs")

# Steps that are basically a code statement
STEP_CODE = re.compile(
    r"^\*\*Шаг\s+\d+\.\*\*\s+`[^`]+`\s*\.?\s*$",
    re.M,
)
# Or: **Шаг N.** something = Something(
STEP_ASSIGN = re.compile(
    r"^\*\*Шаг\s+\d+\.\*\*\s+`?[A-Za-z_][\w.]*\s*=\s*[A-Za-z_]",
    re.M,
)
# Or step that is only a method call in backticks
STEP_CALL = re.compile(
    r"^\*\*Шаг\s+\d+\.\*\*\s+`[A-Za-z_][\w.]*\([^`]*\)`\s*\.?\s*$",
    re.M,
)

# Also catch numbered list items that are pure code
LIST_CODE = re.compile(
    r"^\d+\.\s+`[A-Za-z_][\w.]*\s*=\s*[^`]+`\s*$",
    re.M,
)

files = []
for p in ROOT.rglob("*.ipynb"):
    name = p.name
    if any(
        x in str(p)
        for x in (
            ".ipynb_checkpoints",
            "_legacy",
            "_rankpulse",
            "_school_pass",
            "one_hot",
        )
    ):
        continue
    files.append(p)

print(f"scanning {len(files)} notebooks\n")
for p in sorted(files):
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print("FAIL", p, e)
        continue
    hits = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "markdown":
            continue
        s = "".join(c.get("source", []))
        # only care about assignment-like cells
        if "Задание" not in s and "Что сделать" not in s and "**Шаг" not in s:
            continue
        for rx in (STEP_CODE, STEP_ASSIGN, STEP_CALL, LIST_CODE):
            for m in rx.finditer(s):
                line = m.group(0).strip()
                # skip benign conceptual lines that mention code as example mid-sentence
                if "например" in line.lower():
                    continue
                hits.append((i, line[:160]))
    # dedupe
    seen = set()
    uniq = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        uniq.append(h)
    if uniq:
        print("=" * 80)
        print(p.relative_to(ROOT))
        for i, line in uniq:
            print(f"  cell{i}: {line}")
