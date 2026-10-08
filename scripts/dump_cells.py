import json
import sys
from pathlib import Path

NOTEBOOK = "Laser_Prog_work.ipynb"

if len(sys.argv) < 2:
    print("Usage: python scripts/dump_cells.py <cell_idx> [<cell_idx> ...]")
    sys.exit(1)

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

out = []
for arg in sys.argv[1:]:
    i = int(arg)
    if i < 0 or i >= len(nb["cells"]):
        print(f"# [skipped invalid cell index: {i}]", file=sys.stderr)
        continue
    cell = nb["cells"][i]
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    out.append(f"# ===== CELL [{i}] ({cell['cell_type']}) =====")
    out.append(src)
    out.append("")

Path("dump.txt").write_text("\n".join(out), encoding="utf-8")
print(f"Written {len(out)} chunks to dump.txt")