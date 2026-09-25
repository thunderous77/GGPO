"""Compute the paper's mean@32 and unbiased pass@K from a validation JSONL."""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def summarize(path):
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        groups[(row["data_source"], row["uid"])].append(row["score"])
    per_dataset = defaultdict(list)
    for (source, _), scores in groups.items():
        if len(scores) != 32:
            raise ValueError(f"Expected 32 responses per problem, got {len(scores)} in {source}")
        correct = sum(score == 1 for score in scores)
        per_dataset[source].append([1 - comb(32 - correct, k) / comb(32, k) for k in (1, 4, 8, 16, 32)])
    per_dataset["weighted_average"] = [row for rows in per_dataset.values() for row in rows]
    return {
        source: {
            "problems": len(rows),
            **{f"pass@{k}": 100 * sum(row[i] for row in rows) / len(rows) for i, k in enumerate((1, 4, 8, 16, 32))},
        }
        for source, rows in per_dataset.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="Validation dump (0.jsonl for an exported checkpoint)")
    print(json.dumps(summarize(parser.parse_args().jsonl), indent=2))
