# -*- coding: utf-8 -*-
"""Extract task titles + first question lines for review."""
import json
import re
from pathlib import Path

root = Path(r"c:\Users\Гуровиц Владимир\ml-speckurs")
files = sorted(root.glob("Урок_*_Теория/Тест_к_теории.ipynb"))
out = Path(r"c:\Users\Гуровиц Владимир\ml-speckurs\_tmp_titles_review.txt")

lines_out = []
for f in files:
    nb = json.loads(f.read_text(encoding="utf-8"))
    lines_out.append("=" * 80)
    lines_out.append(str(f.relative_to(root)))
    lines_out.append("=" * 80)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell.get("source", []))
        m = re.match(r"^(#{1,3}\s+.+)$", src.strip().splitlines()[0] if src.strip() else "", re.M)
        first_line = src.strip().splitlines()[0] if src.strip() else ""
        if not re.match(r"^##\s+\d+", first_line):
            continue
        # Print title + next non-empty lines (question body preview)
        body_lines = [ln for ln in src.strip().splitlines()[1:] if ln.strip()][:8]
        lines_out.append(f"\n--- cell[{i}] ---")
        lines_out.append(first_line)
        for bl in body_lines:
            lines_out.append("  | " + bl[:200])

out.write_text("\n".join(lines_out), encoding="utf-8")
print(f"Wrote {out}, {len(lines_out)} lines")
