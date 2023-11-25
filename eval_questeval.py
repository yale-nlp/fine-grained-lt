import argparse
from datasets import load_dataset
from utils_questeval import compute_metrics

# Get dataset from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--preds_path", required=True)
parser.add_argument("--num_samples", required=False, type=int, default=None)
args = parser.parse_args()
print(f"Using dataset: {args.dataset}")
print(f"Using prediction text file: {args.preds_path}")

DATASET_NAME = args.dataset
dataset = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="train"
)
dataset["test"] = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="test"
)["train"]

sources = [item["input"] for item in dataset["test"]]
labels = [item["labels"] for item in dataset["test"]]

with open(args.preds_path) as f:
    preds = f.read().splitlines()

if len(preds) > len(sources):
    preds = list(filter(lambda s: s != "", preds))

if args.num_samples is not None:
    sources = sources[:args.num_samples]
    preds   = preds[:args.num_samples]
    labels  = labels[:args.num_samples]
result = compute_metrics(
    sources,
    preds,
    labels,
    [
        "questeval",
    ],
)
print(result)
print("Copy this string into the Excel and separate by comma")
print(
    ",".join(
        [
            str(result[key])
            for key in [
                "questeval_no_ref",
                "questeval_ref",
            ]
        ]
    )
)
