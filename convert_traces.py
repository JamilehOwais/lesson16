import json
import pandas as pd

spans = []
# Read the exported file line by line
with open("phoenix_10_queries.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            spans.append(json.loads(line))

# Normalize JSON structure to flat table columns
df = pd.json_normalize(spans)

# Save to CSV deliverable
df.to_csv("phoenix_traces.csv", index=False)
print(f"Successfully converted {len(df)} traces into phoenix_traces.csv")