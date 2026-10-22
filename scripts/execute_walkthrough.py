"""Execute the authored walkthrough sequentially; save outputs without Jupyter extras."""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
path = root / "notebooks" / "portfolio_walkthrough.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
count = 0
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    count += 1
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    cell["execution_count"] = count
    cell["outputs"] = [{"output_type":"stream", "name":"stdout", "text":output.getvalue().splitlines(True)}]
path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"Executed {count} trusted authored cells sequentially; no training or holdout reevaluation.")
