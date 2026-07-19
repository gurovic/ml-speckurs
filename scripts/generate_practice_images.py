"""Generate static images for workflow_practice.ipynb from approved.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "11_workflow"
IMG = ROOT / "practice_images"
IMG.mkdir(exist_ok=True)

items = json.loads((ROOT / "approved.json").read_text(encoding="utf-8"))
matrices, labels = [], []
for item in items:
    matrices.append(np.array(item["data"]["pixels"], dtype=int))
    labels.append(item["data"]["initial"])
matrices = np.array(matrices)
labels = np.array(labels)

by_class: dict[str, list[int]] = defaultdict(list)
for i, lbl in enumerate(labels):
    by_class[lbl].append(i)


def save_icon(ax, matrix, title: str) -> None:
    ax.imshow(matrix, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# --- samples per class (3 each) ---
fig, axes = plt.subplots(3, 3, figsize=(5, 5))
class_names = ["castle", "dragon", "knight"]
ru = {"castle": "замок", "dragon": "дракон", "knight": "рыцарь"}
for row, cls in enumerate(class_names):
    for col in range(3):
        idx = by_class[cls][col * 15 + 5]
        save_icon(axes[row, col], matrices[idx], f"{ru[cls]} ({cls})")
fig.suptitle("Примеры иконок по классам (10×10 пикселей)", fontsize=13)
fig.tight_layout()
fig.savefig(IMG / "samples_by_class.png", dpi=150, bbox_inches="tight")
plt.close(fig)

for cls in class_names:
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    idx = by_class[cls][0]
    save_icon(ax, matrices[idx], f"{ru[cls]} — {cls}")
    fig.tight_layout()
    fig.savefig(IMG / f"sample_{cls}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- full dataset mosaic ---
cols = 23
rows_n = int(np.ceil(len(matrices) / cols))
color_map = {"castle": "#4C78A8", "dragon": "#E45756", "knight": "#54A24B"}
fig, ax = plt.subplots(figsize=(14, 6))
canvas = np.ones((rows_n * 12, cols * 12)) * 0.95
for i, (m, lbl) in enumerate(zip(matrices, labels)):
    r, c = divmod(i, cols)
    y0, x0 = r * 12 + 1, c * 12 + 1
    patch = np.where(m, 0.15, 0.85)
    canvas[y0 : y0 + 10, x0 : x0 + 10] = patch
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
ax.set_title(f"Весь датасет: {len(matrices)} иконок (светлые квадраты 10×10)", fontsize=13)
ax.set_xticks([])
ax.set_yticks([])
from matplotlib.patches import Patch

legend = [Patch(facecolor="white", edgecolor="black", label=f"{ru[k]} ({k}): {len(v)}") for k, v in by_class.items()]
ax.legend(handles=legend, loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(IMG / "dataset_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- colored mosaic by class blocks ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, cls in zip(axes, class_names):
    subset = matrices[by_class[cls]]
    n = len(subset)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    grid = np.ones((nrow * 12, ncol * 12)) * 0.95
    for j, m in enumerate(subset):
        r, c = divmod(j, ncol)
        y0, x0 = r * 12 + 1, c * 12 + 1
        grid[y0 : y0 + 10, x0 : x0 + 10] = np.where(m, 0.1, 0.9)
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{ru[cls]} — {n} шт.", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Датасет по классам", fontsize=14)
fig.tight_layout()
fig.savefig(IMG / "dataset_by_class.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- reference charts for Дано ---
counts = {k: len(v) for k, v in by_class.items()}
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(counts.keys(), counts.values(), color=[color_map[k] for k in counts])
ax.set_title("Сколько иконок каждого класса в датасете")
ax.set_xlabel("класс")
ax.set_ylabel("количество")
for x, y in zip(counts.keys(), counts.values()):
    ax.text(x, y + 1, str(y), ha="center")
fig.tight_layout()
fig.savefig(IMG / "chart_class_counts.png", dpi=150, bbox_inches="tight")
plt.close(fig)

density = matrices.reshape(len(matrices), -1).mean(axis=1)
fig, ax = plt.subplots(figsize=(6, 3.5))
data = [density[labels == cls] for cls in class_names]
ax.boxplot(data, labels=class_names)
ax.set_title("Плотность пикселей (доля единиц) по классам")
ax.set_ylabel("pixel_density")
fig.tight_layout()
fig.savefig(IMG / "chart_density_boxplot.png", dpi=150, bbox_inches="tight")
plt.close(fig)

bottom = matrices[:, -1, :].sum(axis=1)
fig, ax = plt.subplots(figsize=(6, 3.5))
means = [bottom[labels == cls].mean() for cls in class_names]
ax.bar(class_names, means, color=[color_map[k] for k in class_names])
ax.set_title("Среднее число единиц в нижнем ряду (10 пикселей)")
ax.set_xlabel("класс")
ax.set_ylabel("среднее")
fig.tight_layout()
fig.savefig(IMG / "chart_bottom_row.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("saved to", IMG)
for p in sorted(IMG.glob("*.png")):
    print(" ", p.name)
