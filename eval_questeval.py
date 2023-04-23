from questeval.questeval_metric import QuestEval
import argparse
import numpy as np
import json

questeval = QuestEval(no_cuda=False)


def compute_metrics(sources, predictions, labels):
    """_summary_

    Args:
        source (list[str]): List of input sources
        prediction (list[str]): List of output sources
        labels (list[list[str]]): List of list of reference strings

    Returns:
        dict: Output of computed metrics
    """
    result = {}

    score = questeval.corpus_questeval(
        hypothesis=predictions, sources=sources, list_references=labels
    )
    result["questeval_ref"] = score["corpus_score"]
    result["questeval_ref_std"] = np.std(score["ex_level_scores"])

    score = questeval.corpus_questeval(
        hypothesis=predictions,
        sources=sources,
    )
    result["questeval_no_ref"] = score["corpus_score"]
    result["questeval_no_ref_std"] = np.std(score["ex_level_scores"])

    return result


# Get dataset from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--preds_path", required=True)
args = parser.parse_args()
print(f"Using dataset: {args.dataset}")
print(f"Using prediction text file: {args.preds_path}")

DATASET_NAME = args.dataset
dataset = json.load(open(f"data/{DATASET_NAME}_multiple.json", "r"))

sources = [item["input"] for item in dataset["test"]]
labels = [item["labels"] for item in dataset["test"]]

with open(args.preds_path) as f:
    preds = f.read().splitlines()

print(compute_metrics(sources, preds, labels))
