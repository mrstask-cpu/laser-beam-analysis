import json
from pathlib import Path

NOTEBOOK = "Laser_Prog_work.ipynb"
OUTPUT = "notebook_map.txt"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

lines = []
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    first = next((l for l in src.split("\n") if l.strip()), "")
    kind = "MD  " if cell["cell_type"] == "markdown" else "CODE"
    lines.append(f"[{i:>3}] {kind}: {first[:120]}")

    if cell["cell_type"] == "code":
        for line in src.split("\n"):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                lines.append(f"        -> {stripped[:120]}")

Path(OUTPUT).write_text("\n".join(lines), encoding="utf-8")

print(f"Written {len(lines)} lines to {OUTPUT}")
print(f"Total cells: {len(nb['cells'])}")