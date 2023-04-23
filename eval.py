import pandas as pd
import numpy as np
import nltk
from datasets import Dataset, DatasetDict, load_metric, load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load
import argparse
from utils import get_readability_score
from collections import Counter

metric_rouge = load("rouge")
metric_bertscore = load("bertscore")
metric_sari = load("sari")


def compute_metrics(sources, predictions, labels):
    """_summary_

    Args:
        source (list[str]): List of input sources
        prediction (list[str]): List of output sources
        labels (list[list[str]]): List of list of reference strings

    Returns:
        dict: Output of computed metrics
    """
    result_rouge = metric_rouge.compute(
        predictions=predictions, references=labels, use_stemmer=True
    )
    result_sari = metric_sari.compute(
        sources=sources, predictions=predictions, references=labels
    )

    result_bert = []
    for (pred, label) in zip(predictions, labels):
        result_bert_temp = metric_bertscore.compute(
            predictions=[pred] * len(label), references=label, lang="en"
        )
        result_bert.append(result_bert_temp["f1"])

    READABILITY_METRICS = ["flesch_reading_ease"]
    readability_dict = {}
    for metric in READABILITY_METRICS:
        result_readability = list(
            map(lambda s: get_readability_score(s, metric=metric), predictions)
        )
        readability_dict[f"{metric}_counts"] = Counter(
            list(map(lambda item: item[1], result_readability))
        )
        readability_dict[f"{metric}_score"] = np.mean(
            list(map(lambda item: item[0], result_readability))
        )

    # Extract results
    result = result_rouge
    result["bert_score"] = np.mean(result_bert)
    result["sari"] = result_sari["sari"]
    result.update(readability_dict)

    return {k: round(v, 4) if "counts" not in k else v for k, v in result.items()}


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

print(compute_metrics(sources, preds, labels))
