import json, sys
from pathlib import Path

p = Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
print("Articles:", len(data))
print("First 2 items preview:")
for x in data[:2]:
    print("-", x["id"], "|", x["title"][:60], "...")
    print("  ", x["text"][:120].replace("\n"," ") + "...")
