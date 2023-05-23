import argparse
import numpy as np
import pandas as pd
import pickle
import spacy
import torch
import wandb

from datasets import load_dataset, Dataset
from nltk.tokenize import word_tokenize
from scispacy.linking import EntityLinker
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from typing import Any, Dict, List, Optional, Tuple, Union
from utils_eval import get_readability_score, clean_string

# Get dataset from arguments
parser = argparse.ArgumentParser()

# Dataset and model
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
# parser.add_argument("--gpu_id", required=True, type=str)
parser.add_argument("--checkpoint", required=False, type=str, default=None)

# Generation mode
parser.add_argument("--pred_split_sent", required=False, type=str, default="False")

args = parser.parse_args()

device = torch.device(f"cuda")

ner_model = spacy.load("en_core_sci_lg")
ner_model.add_pipe(
    "scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}
)
linker = ner_model.get_pipe("scispacy_linker")

SEMTYPES = [
    "T023",
    "T028",
    "T047",
    "T048",
    "T060",
    "T061",
    "T074",
    "T109",
    "T116",
    "T121",
    "T122",
    "T123",
    "T125",
    "T129",
    "T184",
    "T191",
    "T195",
]

# Load in the model and tokenizer, for this we're using BART,
# which is good at generation tasks
model_name_dict = {
    "bart": ("BART", "facebook/bart-large"),
    "bart_xsum": ("BART_XSUM", "facebook/bart-large-xsum"),
    "flant5": ("FLANT5_LARGE", "google/flan-t5-large"),
    "flant5_base": ("FLANT5_BASE", "google/flan-t5-base"),
}


def get_semtype(e):
    if len(e._.kb_ents) > 0:
        umls_ent_id, score = e._.kb_ents[0]  # Get top hit from UMLS
        umls_ent = linker.kb.cui_to_entity[umls_ent_id]  # Get UMLS entity
        umls_semt = umls_ent[3]
        return umls_semt
    else:
        return []


def split_sent(s):
    input_dict = tokenizer(s)
    num_toks = len(input_dict["input_ids"])
    if num_toks <= 768:
        return [s]
    elif num_toks <= 768 * 2:
        lst = s.split(". ")
        first = ". ".join(lst[: len(lst) // 2])
        second = ". ".join(lst[len(lst) // 2 :])
        return [first, second]
    elif num_toks <= 768 * 3:
        lst = s.split(". ")
        first = ". ".join(lst[: len(lst) // 3])
        second = ". ".join(lst[len(lst) // 3 : 2 * len(lst) // 3])
        third = ". ".join(lst[2 * len(lst) // 3 :])
        return [first, second, third]
    else:
        lst = s.split(". ")
        first = ". ".join(lst[: len(lst) // 4])
        second = ". ".join(lst[len(lst) // 4 : 2 * len(lst) // 4])
        third = ". ".join(lst[2 * len(lst) // 4 : 3 * len(lst) // 4])
        fourth = ". ".join(lst[3 * len(lst) // 4 :])
        return [first, second, third, fourth]


def generate(dataloader):

    result_base, result_entity, result_complexity = [], [], []

    model.eval()
    for i, batch in enumerate(dataloader):

        print(f"On {i} of {len(dataloader)}")

        words = word_tokenize(batch["input"][0])
        scores = [
            max(get_readability_score(t, metric="flesch_kincaid_grade")[0], 0.0)
            for t in words
        ]
        exclude_words = [tup[0] for tup in zip(words, scores) if tup[1] > 10]
        if len(exclude_words) > 0:
            bad_words_com = tokenizer(exclude_words, add_special_tokens=False)[
                "input_ids"
            ]
        else:
            bad_words_com = None

        doc = ner_model(batch["input"][0])
        ents = doc.ents
        exclude_ents = [
            str(e) for e in ents if any([s in SEMTYPES for s in get_semtype(e)])
        ]
        if len(exclude_ents) > 0:
            bad_words_ent = tokenizer(exclude_ents, add_special_tokens=False)[
                "input_ids"
            ]
        else:
            bad_words_ent = None

        model_inputs = tokenizer(
            batch["input"],
            max_length=768,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs_com = model.generate(
            input_ids=model_inputs["input_ids"],
            max_length=768,
            bad_words_ids=bad_words_com,
        )
        outputs_ent = model.generate(
            input_ids=model_inputs["input_ids"],
            max_length=768,
            bad_words_ids=bad_words_ent,
        )
        outputs_bas = model.generate(
            input_ids=model_inputs["input_ids"], max_length=768, bad_words_ids=None
        )
        result_complexity.extend(tokenizer.batch_decode(outputs_com))
        result_entity.extend(tokenizer.batch_decode(outputs_ent))
        result_base.extend(tokenizer.batch_decode(outputs_bas))
        print(tokenizer.batch_decode(outputs_bas))

    return result_base, result_entity, result_complexity


# Naming variables
tokenizer = AutoTokenizer.from_pretrained(
    model_name_dict[args.model][1], add_prefix_space=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    args.checkpoint if args.checkpoint is not None else model_name_dict[args.model][1]
).to(device)

PROJECT_NAME = f"{args.dataset}_{model_name_dict[args.model][0]}"

# Load and preprocess the datasets
dataset = load_dataset("json", data_files=f"data/{args.dataset}.json", field="train")
dataset["test"] = load_dataset(
    "json", data_files=f"data/{args.dataset}_multiple.json", field="test"
)["train"]

if args.pred_split_sent == "True":
    df = pd.DataFrame(dataset["test"])
    input_sents = list(map(split_sent, df["input"]))
    report_id_lst = []
    input_sent_lst = []
    for idx, lst in enumerate(input_sents):
        report_id_lst.extend([idx] * len(lst))
        input_sent_lst.extend(lst)
    df = pd.DataFrame({"report_id": report_id_lst, "input": input_sent_lst})
    df["labels"] = [[""]] * len(df)
    dataset["test"] = Dataset.from_pandas(df[["input", "labels"]])

dataloader = DataLoader(dataset["test"], batch_size=1)

result_base, result_entity, result_complexity = generate(dataloader)

result_complexity = list(map(clean_string, result_complexity))
result_entity = list(map(clean_string, result_entity))
result_base = list(map(clean_string, result_base))

# # Collate sentences by report
# if pred_split_sent:
#     df["prediction"] = test_output
#     df = df.groupby("report_id")\
#             .aggregate({"prediction": lambda lst: ". ".join(lst)})\
#             .reset_index(drop=True)
#     test_output = list(df["prediction"])

# Write output
with open(f"output/{PROJECT_NAME}_complexity.txt", "w") as fp:
    for item in result_complexity:
        fp.write("%s\n" % item)

with open(f"output/{PROJECT_NAME}_entity.txt", "w") as fp:
    for item in result_entity:
        fp.write("%s\n" % item)

with open(f"output/{PROJECT_NAME}_base.txt", "w") as fp:
    for item in result_base:
        fp.write("%s\n" % item)
