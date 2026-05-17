from datasets import load_dataset
import json

# 1 load data
# Company + Question + Answer + Evidence
ds = load_dataset("PatronusAI/financebench")

# 2 create documents
docs = [
    f"""
Company: {item['company']}
Question: {item['question']}
Answer: {item['answer']}
Evidence: {item['evidence']}
""".strip()

    for item in ds["train"]
]

# 3 save dataset
with open("financebenchDataset.txt", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)