from datasets import load_dataset
import numpy as np
import json

# Download Alpaca 52K
dataset = load_dataset("tatsu-lab/alpaca")["train"]
print(f"Total examples in Alpaca 52K: {len(dataset)}")

# Randomly select samples from Alpaca at 10% budget
budget = 0.1
num_samples = int(len(dataset) * budget)
rng = np.random.default_rng(seed=42)  # For reproducibility
selected_indices = sorted(rng.choice(len(dataset), size=num_samples, replace=False).tolist())

selection = {
    "method": "random",
    "budget_fraction": budget,
    "selected_samples": selected_indices,
    "total_samples": len(dataset),
    "seed": 42,
    "dataset_name": "tatsu-lab/alpaca"
}

with open("random_10pct.json", "w") as f:
    json.dump(selection, f)

print("Saved random selection to random_10pct.json")