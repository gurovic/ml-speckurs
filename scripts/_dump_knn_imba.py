"""Temp: dump knn_imba.ipynb cell sources for review."""
import json
from pathlib import Path

src_path = Path(r"C:\Users\Гуровиц Владимир\Downloads\knn_imba.ipynb")
dst_path = Path(r"C:\Users\Гуровиц Владимир\Downloads\knn_imba_cells.txt")

nb = json.loads(src_path.read_text(encoding="utf-8"))
out = [f"cells: {len(nb['cells'])}", ""]
for i, c in enumerate(nb["cells"]):
    text = "".join(c.get("source", []))
    out.append(f"=== cell {i} [{c['cell_type']}] ===")
    out.append(text[:800])
    out.append("")
dst_path.write_text("\n".join(out), encoding="utf-8")
print("written", dst_path)
