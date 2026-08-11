# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path

root = Path(r"c:\Users\Гуровиц Владимир\ml-speckurs")
files = sorted(root.glob("Урок_*_Теория/Тест_к_теории.ipynb"))

hint_pat = re.compile(
    r"(ответ|верно|правильн|потому что|так как|StandardScaler|MinMaxScaler|"
    r"validation|overfitting|underfitting|\(ответ|\(верно|→|"
    r"выбери\s+\w+|нужно\s+использовать|следует\s+использовать|"
    r"не\s+надо|нельзя\s+|должен\s+быть|должна\s+быть)",
    re.I,
)

for f in files:
    nb = json.loads(f.read_text(encoding="utf-8"))
    print("=" * 80)
    print(f.relative_to(root))
    print("=" * 80)
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        lines = src.splitlines()
        headings = []
        for line in lines[:20]:
            if re.match(r"^#{1,4}\s+", line):
                headings.append(line)
            elif headings and line.strip() == "":
                continue
            elif headings:
                break
        first_md = None
        if cell["cell_type"] == "markdown":
            for line in lines:
                if line.strip():
                    first_md = line.strip()
                    break
        text = " | ".join(headings) if headings else first_md
        if not text:
            continue
        is_taskish = bool(
            headings
            or re.search(
                r"Задани|Задач|Вопрос|Тест\s*\d|#+\s*\d|Задача",
                text or "",
                re.I,
            )
        )
        if not is_taskish:
            continue
        flag = " *** HINT? ***" if hint_pat.search(text or "") else ""
        ctype = cell["cell_type"]
        print(f"  cell[{i}] {ctype}: {text}{flag}")
