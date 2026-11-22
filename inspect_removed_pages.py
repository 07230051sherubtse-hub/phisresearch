import pandas as pd

d = pd.read_csv("phishing_papers/pages_cleaned.csv")

samples = [
    ("A013", [13, 16]),
    ("A034", [2, 8]),
    ("A061", [9, 24]),
    ("A204", [4, 19]),
    ("A243", [21, 22]),
    ("B025", [5, 26]),
    ("C021", [3, 37])
]

for pid, pages in samples:
    print("\n" + "=" * 70)
    print("ARTICLE:", pid)

    for page in pages:
        result = d[
            (d["Article ID"] == pid) &
            (d["Page"] == page)
        ]

        if not result.empty:
            print("\nPAGE:", page)
            print(str(result.iloc[0]["Text"])[:900])
