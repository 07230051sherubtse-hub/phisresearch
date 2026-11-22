import json

with open("phishing_papers/Untitled.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

keywords = [
    "filter", "drop(", "Word_Count", "Text_Length",
    "reference", "pages", "cleaned", "remove"
]

for i, cell in enumerate(notebook["cells"]):
    source = "".join(cell.get("source", []))
    if cell["cell_type"] == "code" and any(k.lower() in source.lower() for k in keywords):
        print("\n" + "=" * 60)
        print("CELL:", i)
        print(source[:3500])


