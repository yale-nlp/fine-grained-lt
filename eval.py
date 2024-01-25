import argparse
from datasets import load_dataset
from utils.utils_eval import compute_metrics

# Get dataset from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--preds_path", required=True)
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

result = compute_metrics(
    sources,
    preds,
    labels,
    [
        "rouge",
        "bert_score",
        # "bert_score_l",
        # "sari",
        "sari_easse",
        # "flesch_kincaid_grade",
        "fkgl_easse",
        "ari",
        "check_entities"
    ],
)
print(result)
print("Copy this string into the Excel and separate by comma")
print(
    ",".join(
        [
            str(result[key])
            for key in [
                "check_entities",
                "fkgl_easse",
                "ari_score",
                "bert_score",
                "sari_easse",
                "rougeLsum",
                # "rouge1",
                # "rouge2",
                "rougeL",
                # "bert_score_l",
                # "sari",
                # "flesch_kincaid_grade_score",
            ]
        ]
    )
)
