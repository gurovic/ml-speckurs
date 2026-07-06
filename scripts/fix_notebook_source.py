"""Fix notebook cells where source was split character-by-character."""
import json
import sys
from pathlib import Path


def normalize_source(text: str) -> list[str]:
    if not text:
        return [""]
    lines = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        if lines and lines[-1].endswith("\n"):
            pass
        elif not lines:
            return [text]
        else:
            # splitlines drops trailing empty line but keeps content without \n on last line
            return lines
    return lines if lines else [""]


def is_char_split(src: list[str]) -> bool:
    if len(src) <= 50:
        return False
    return all(len(s) <= 2 for s in src[:30])


def fix_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    fixed = 0
    for i, cell in enumerate(nb["cells"]):
        src = cell.get("source", [])
        if is_char_split(src):
            text = "".join(src)
            cell["source"] = normalize_source(text)
            fixed += 1
            print(
                f"Fixed cell {i} ({cell['cell_type']}): "
                f"{len(src)} fragments -> {len(cell['source'])} lines"
            )
    if fixed:
        path.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return fixed


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not target:
        sys.exit("Usage: fix_notebook_source.py <notebook.ipynb>")
    print(f"Fixing {target}")
    print(f"Total fixed: {fix_notebook(target)}")
