import pandas as pd
import numpy as np
import nltk

# nltk.download('punkt')
from datasets import Dataset, DatasetDict, load_metric, load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load
import argparse
import textstat

# from utils import get_readability_score
from collections import Counter

metric_rouge = load("rouge")
metric_bertscore = load("bertscore")
metric_sari = load("sari")


def get_readability_score(text, metric="flesch_reading_ease"):
    """get the readability score and grade level of text"""
    if metric == "flesch_reading_ease":
        score = textstat.flesch_reading_ease(text)
        if score > 90:
            grade = "5th grade"
        elif score > 80:
            grade = "6th grade"
        elif score > 70:
            grade = "7th grade"
        elif score > 60:
            grade = "8th & 9th grade"
        elif score > 50:
            grade = "10th to 12th grade"
        elif score > 30:
            grade = "college"  # Collge student = average 13th to 15th grade
        elif score > 10:
            grade = "college graduate"
        else:
            grade = "professional"
        return score, grade

    elif metric == "flesch_kincaid_grade":
        score = textstat.flesch_kincaid_grade(
            text
        )  # Note: this score can be negative like -1
        grade = round(score)
        if grade > 16:
            grade = "college graduate"  # Collge graduate: >16th grade
        elif grade > 12:
            grade = "college"
        elif grade <= 4:
            grade = "4th grade or lower"
        else:
            grade = f"{grade}th grade"
        return score, grade

    # elif metric == 'SMOG': # Note: SMOG formula needs at least three ten-sentence-long samples for valid calculation
    #     score = textstat.smog_index(text)
    #     grade = round(score)
    #     if grade > 16:
    #         grade = 'college graduate'
    #     elif grade > 12:
    #         grade = 'college'
    #     else:
    #         grade = f"{grade}th grade"
    #     return score, grade

    elif metric == "dale_chall":
        score = textstat.dale_chall_readability_score(text)
        if score >= 10:
            grade = "college graduate"
        elif score >= 9:
            grade = "college"  # Collge student = average 13th to 15th grade
        elif score >= 8:
            grade = "11th to 12th grade"
        elif score >= 7:
            grade = "9th to 10th grade"
        elif score >= 6:
            grade = "7th to 8th grade"
        elif score >= 5:
            grade = "5th to 6th grade"
        else:
            grade = "4th grade or lower"
        return score, grade

    elif metric == "gunning_fog":
        score = textstat.gunning_fog(text)
        grade = round(score)
        if grade > 16:
            grade = "college graduate"
        elif grade > 12:
            grade = "college"
        elif grade <= 4:
            grade = "4th grade or lower"
        else:
            grade = f"{grade}th grade"
        return score, grade

    else:
        raise ValueError(f"Unknown metric {metric}")


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
