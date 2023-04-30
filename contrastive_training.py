import argparse
import numpy as np
import os
import pandas as pd
import random
from tqdm.auto import tqdm
import wandb

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AdamW,
    get_scheduler,
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device("cuda")

# Get dataset from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kg_path", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--lr", required=False, type=float, default=1e-5)
parser.add_argument("--epochs", required=False, type=int, default=5)
parser.add_argument("--batch_size", required=False, type=int, default=1)
parser.add_argument("--checkpoint", required=False, type=str, default=None)
parser.add_argument("--weight_decay", required=False, type=float, default=0.01)
parser.add_argument("--warmup_steps", required=False, type=int, default=1000)
parser.add_argument(
    "--gradient_accumulation_steps", required=False, type=int, default=1
)
args = parser.parse_args()

# Initialize logging in WANDB
run = wandb.init(
    project=f"contrastive_pretraining_{args.model}",
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "model": args.model,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
    },
)

# Load knowledge graph definitions
PATH = f"/home/lily/lyf6/Simplification-Project/{args.kg_path}"
KG_NAME = PATH.split("/")[-1]
filenames = os.listdir(PATH)

terms = [i[:-4].replace("_", " ") for i in filenames]
descs = [" ".join([line.strip() for line in open(f"{PATH}/{f}")]) for f in filenames]
labels = [1.0] * len(terms)
num_terms = len(terms)

# Create negative examples
label_lst = []
term_lst = []
desc_lst = []

shift_idxs = random.sample(range(1, num_terms), k=3)

for shift in shift_idxs:
    new_idxs = (np.arange(num_terms) + shift) % num_terms
    term_lst.extend(terms)
    desc_lst.extend([descs[i] for i in new_idxs])
    label_lst.extend([0.0] * num_terms)

# Create dataframe, dataset, and dataloader
df = pd.DataFrame(
    {
        "terms": terms + term_lst,
        "definitions": descs + desc_lst,
        "labels": labels + label_lst,
    }
)
df = df.dropna().reset_index(drop=True)

dataset = Dataset.from_pandas(df)
dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

loss_function = torch.nn.BCELoss()

# Load model and tokenizer
if args.model == "bart":
    model_path = "facebook/bart-large"
    MODEL_NAME = "BART"
elif args.model == "flant5":
    model_path = "google/flan-t5-large"
    MODEL_NAME = "FLANT5_LARGE"
elif args.model == "flant5_base":
    model_path = "google/flan-t5-base"
    MODEL_NAME = "FLANT5_BASE"
else:
    assert False

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path if args.checkpoint == None else args.checkpoint,
    output_hidden_states=True,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=args.lr)

num_training_steps = args.epochs * len(dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=num_training_steps,
)


def preprocess_function(examples):
    """ """
    model_terms = tokenizer(
        examples["terms"],
        max_length=100,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    model_terms["labels"] = model_terms["input_ids"]

    model_defns = tokenizer(
        examples["definitions"],
        max_length=100,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    model_defns["labels"] = model_defns["input_ids"]

    return model_terms, model_defns


model.train()

progress_bar = tqdm(range(num_training_steps))

for ep in range(args.epochs):
    for idx, batch in enumerate(dataloader):

        optimizer.zero_grad()

        batch_terms, batch_defns = preprocess_function(batch)
        batch_terms = {k: v.to(device) for k, v in batch_terms.items()}
        batch_defns = {k: v.to(device) for k, v in batch_defns.items()}
        labels = batch["labels"].to(torch.float).to(device)

        # Get model embeddings
        x = model(**(batch_terms + batch_defns))  # TODO: Lj -
        y = model(**batch_defns)

        # Get mean embedding only for actual entries
        x_actual = (
            batch_terms["attention_mask"].unsqueeze(2) * x["encoder_last_hidden_state"]
        )
        y_actual = (
            batch_defns["attention_mask"].unsqueeze(2) * y["encoder_last_hidden_state"]
        )

        x_num_toks = batch_terms["attention_mask"].sum(axis=1).unsqueeze(1)
        y_num_toks = batch_defns["attention_mask"].sum(axis=1).unsqueeze(1)

        x_clean = x_actual.sum(axis=1) / x_num_toks
        y_clean = y_actual.sum(axis=1) / y_num_toks

        # Compute cosine similarity of mean embeddings
        output = F.cosine_similarity(x_clean, y_clean)

        # Rescale cosine similarity outputs
        output = (output + 1.0) / 2.0

        # Compute loss
        loss = loss_function(output, labels)

        # Normalize and backprop the gradients
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        wandb.log({"loss": loss})

        # Gradient accumulation then step
        if ((idx + 1) % args.gradient_accumulation_steps == 0) or (
            idx + 1 == len(dataloader)
        ):
            optimizer.zero_grad()
            optimizer.step()
            progress_bar.update(1)


model.save_pretrained(f"models/PRETRAIN_{MODEL_NAME}_{KG_NAME}", from_pt=True)
