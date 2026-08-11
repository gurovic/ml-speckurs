# -*- coding: utf-8 -*-
"""Smoke-test practice 34 without blocking Gradio launch."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

NB = Path(__file__).resolve().parent / "Урок_34_Ансамбли_Bagging_Случайный_лес_Практика.ipynb"


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    print("cells", len(nb["cells"]))
    ns: dict = {"display": print}

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "demo.launch" in src:
            src = src.replace(
                "demo.launch(share=False)",
                'print("GRADIO_UI_BUILT", type(demo))  # demo.launch(share=False)',
            )
        print("--- exec cell", i, "---")
        exec(compile(src, f"cell_{i}", "exec"), ns, ns)

    print("ABLATION val", ns["ablation_df"]["val_acc"].tolist())
    print("PROTOCOL rows", len(ns["experiments_log"]))
    print("TOP features", list(ns["imp_df"]["feature"].head(3)))
    print("OOB", round(float(ns["oob"]), 3), "VAL", round(float(ns["val"]), 3))

    settings, metrics, fig, *rest = ns["train_ablation_config"](
        "random forest", 50, "без ограничения (None)", 8, "sqrt", True
    )
    print("UI settings:", settings)
    print("UI metrics:", metrics.replace("\n", " | "))
    log, msg = ns["add_to_protocol"](
        "smoke hyp",
        "smoke out",
        rest[0],
        rest[1],
        rest[2],
        rest[3],
        rest[4],
        rest[5],
        rest[6],
    )
    print(msg, "log len", len(log))

    # score table check
    text0 = "".join(nb["cells"][0]["source"])
    assert "**30**" in text0
    assert "experiments_log" in "".join("".join(c["source"]) for c in nb["cells"])
    assert "gradio" in "".join("".join(c["source"]) for c in nb["cells"]).lower()
    print("OK")


if __name__ == "__main__":
    main()
