import argparse
import pandas as pd
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
parser.add_argument("--kg", required=True)
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
parser.add_argument("--loss_type", required=False, type=str, default="cs")
args = parser.parse_args()

# Initialize logging in WANDB
run = wandb.init(
    project=f"contrastive_pretraining_{args.model}_{args.kg}_{args.loss_type}",
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "model": args.model,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "loss_type": args.loss_type,
    },
)

df = pd.read_csv(
    f"data/contrastive/{args.kg}_{args.loss_type}.csv", lineterminator="\n"
)
df = df.dropna().reset_index(drop=True)

dataset = Dataset.from_pandas(df)
dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

if args.loss_type == "cs":
    criterion = torch.nn.BCELoss()
elif args.loss_type in ["mse_minimize", "mse_contrastive"]:
    criterion = torch.nn.MSELoss()
else:
    assert False, print("loss type must be either cs or mse")

model_name_dict = {
    "bart": ("BART", "facebook/bart-large"),
    "bart_xsum": ("BART_XSUM", "facebook/bart-large-xsum"),
    "flant5": ("FLANT5_LARGE", "google/flan-t5-large"),
    "flant5_base": ("FLANT5_BASE", "google/flan-t5-base"),
}

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_dict[args.model][1] if args.checkpoint == None else args.checkpoint,
    output_hidden_states=True,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_dict[args.model][1] if args.checkpoint == None else args.checkpoint
)

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
    inputs = tokenizer(
        examples["terms"] + examples["definitions"] + examples["contrast"]
        if "contrast" in examples.keys()
        else examples["terms"] + examples["definitions"],
        max_length=300,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    num_items = len(examples["terms"])

    model_terms = {}
    model_terms["input_ids"] = inputs["input_ids"][:num_items]
    model_terms["attention_mask"] = inputs["attention_mask"][:num_items]
    model_terms["labels"] = inputs["input_ids"][:num_items]

    model_defns = {}
    model_defns["input_ids"] = inputs["input_ids"][num_items : 2 * num_items]
    model_defns["attention_mask"] = inputs["attention_mask"][num_items : 2 * num_items]
    model_defns["labels"] = inputs["input_ids"][num_items : 2 * num_items]

    model_conts = {}
    if "contrast" in examples.keys():
        model_conts["input_ids"] = inputs["input_ids"][2 * num_items :]
        model_conts["attention_mask"] = inputs["attention_mask"][2 * num_items :]
        model_conts["labels"] = inputs["input_ids"][2 * num_items :]

    return model_terms, model_defns, model_conts


model.train()

progress_bar = tqdm(range(num_training_steps))

for ep in range(args.epochs):
    for idx, batch in enumerate(dataloader):

        optimizer.zero_grad()

        batch_terms, batch_defns, batch_conts = preprocess_function(batch)
        batch_terms = {k: v.to(device) for k, v in batch_terms.items()}
        batch_defns = {k: v.to(device) for k, v in batch_defns.items()}
        batch_conts = {k: v.to(device) for k, v in batch_conts.items()}

        # Get model embeddings
        x = model(**batch_terms)
        y = model(**batch_defns)

        if args.loss_type == "cs":
            # Get labels
            labels = batch["labels"].to(torch.float).to(device)

            # Get mean embedding only for actual entries
            x_actual = (
                batch_terms["attention_mask"].unsqueeze(2)
                * x["encoder_last_hidden_state"]
            )
            y_actual = (
                batch_defns["attention_mask"].unsqueeze(2)
                * y["encoder_last_hidden_state"]
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
            loss = criterion(output, labels)

        if args.loss_type == "mse_minimize":
            x_flatten = (
                batch_terms["attention_mask"].unsqueeze(2)
                * x["encoder_last_hidden_state"]
            ).view(-1)
            y_flatten = (
                batch_defns["attention_mask"].unsqueeze(2)
                * y["encoder_last_hidden_state"]
            ).view(-1)
            loss = criterion(x_flatten, y_flatten)

        if args.loss_type == "mse_contrastive":
            z = model(**batch_conts)

            x_flatten = (
                batch_terms["attention_mask"].unsqueeze(2)
                * x["encoder_last_hidden_state"]
            ).view(-1)
            y_flatten = (
                batch_defns["attention_mask"].unsqueeze(2)
                * y["encoder_last_hidden_state"]
            ).view(-1)
            z_flatten = (
                batch_conts["attention_mask"].unsqueeze(2)
                * z["encoder_last_hidden_state"]
            ).view(-1)

            loss = criterion(x_flatten, y_flatten) - criterion(x_flatten, z_flatten)

        # Normalize and backprop the gradients
        loss.backward()
        wandb.log({"loss": loss})

        # Gradient accumulation then step
        if ((idx + 1) % args.gradient_accumulation_steps == 0) or (
            idx + 1 == len(dataloader)
        ):
            optimizer.zero_grad()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

KG_NAME = str(args.kg).upper()
LOSS_NAME = str(args.loss_type).upper()
model.save_pretrained(
    f"models/PRETRAIN_{model_name_dict[args.model][0]}_{KG_NAME}_{LOSS_NAME}",
    from_pt=True,
)
