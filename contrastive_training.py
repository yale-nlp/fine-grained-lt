import pandas as pd
import numpy as np
import nltk
from tqdm.auto import tqdm

from datasets import Dataset, DatasetDict, load_metric, load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    AdamW,
    get_scheduler,
)
from evaluate import load
import argparse
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

lr = 5e-5
num_epochs = 3

device = torch.device("cuda")

PATH = "data/wordnet_wikipedia"
filenames = os.listdir(PATH)

terms = [i[:-4].replace("_", " ") for i in filenames]
descs = [" ".join([line.strip() for line in open(f"{PATH}/{f}")]) for f in filenames]
labels = [1] * len(terms)
num_terms = len(terms)

label_lst = []
term_lst = []
desc_lst = []

shift_idxs = random.sample(range(num_terms), k=3)

for shift in shift_idxs:
    new_idxs = (np.arange(num_terms) + shift) % num_terms
    term_lst.extend(terms)
    desc_lst.extend([descs[i] for i in new_idxs])
    label_lst.extend([0] * num_terms)

df = pd.DataFrame(
    {
        "terms": terms + term_lst,
        "definitions": descs + desc_lst,
        "labels": labels + label_lst,
    }
)
dataset = Dataset.from_pandas(df)
dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

loss_function = torch.nn.BCELoss()

MODEL_NAME = "FLANT5_BASE"
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base", output_hidden_states=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


optimizer = AdamW(model.parameters(), lr=lr)

num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


def preprocess_function(examples):
    """ """
    model_terms = tokenizer(
        examples["terms"],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    model_terms["labels"] = model_terms["input_ids"]

    model_defns = tokenizer(
        examples["definitions"],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    model_defns["labels"] = model_defns["input_ids"]

    return model_terms, model_defns


model.train()

progress_bar = tqdm(range(num_training_steps))

for batch in dataloader:
    batch_terms, batch_defns = preprocess_function(batch)
    batch_terms = {k: v.to(device) for k, v in batch_terms.items()}
    batch_defns = {k: v.to(device) for k, v in batch_defns.items()}
    labels = batch["labels"].to(torch.float).to(device)

    x = model(**batch_terms)
    y = model(**batch_defns)

    output = F.cosine_similarity(
        x["encoder_last_hidden_state"].mean(axis=1),
        y["encoder_last_hidden_state"].mean(axis=1),
    )

    loss = loss_function(output, labels)

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
