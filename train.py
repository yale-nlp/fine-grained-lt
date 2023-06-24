import argparse
import itertools
import numpy as np
import os
import pandas as pd
import pickle
import spacy
import torch
import wandb

from datasets import load_dataset, Dataset
from nltk.tokenize import sent_tokenize
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    EarlyStoppingCallback
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from typing import Any, Dict, List, Optional, Tuple, Union
from utils_eval import compute_metrics

# Get dataset from arguments
parser = argparse.ArgumentParser()

# Dataset and model
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--checkpoint", required=False, type=str, default=None)

# Naming
parser.add_argument("--suffix", required=False, type=str, default="")

# Script mode
parser.add_argument("--hyperparameter_tune", required=False, type=str, default="False")
parser.add_argument("--hyperparameter_trials", required=False, type=int, default=5)
parser.add_argument("--predict_only", required=False, type=str, default="False")
parser.add_argument("--pred_split_sent", required=False, type=str, default="False")

# Training parameters
parser.add_argument("--learning_rate", required=False, type=float, default=5e-5)
parser.add_argument("--epochs", required=False, type=int, default=1)
parser.add_argument("--batch_size", required=False, type=int, default=1)
parser.add_argument("--weight_decay", required=False, type=float, default=0.01)
parser.add_argument("--loss_type", required=False, type=str, default="standard")
parser.add_argument(
    "--gradient_accumulation_steps", required=False, type=int, default=1
)
parser.add_argument("--warmup_steps", required=False, type=int, default=0)
args = parser.parse_args()

assert args.hyperparameter_tune in ["True", "False"]
assert args.predict_only in ["True", "False"]

lambda_read  = 0.00075
lambda_const = 0.00015

# Load in the model and tokenizer, for this we're using BART,
# which is good at generation tasks
model_name_dict = {
    "bart": ("BART", "facebook/bart-large"),
    "bart_xsum": ("BART_XSUM", "facebook/bart-large-xsum"),
    "flant5": ("FLANT5_LARGE", "google/flan-t5-large"),
    "flant5_base": ("FLANT5_BASE", "google/flan-t5-base"),
}

class SimplificationTrainer(Seq2SeqTrainer):
    def shift_tokens_right(self, input_ids, pad_token_id):
        """Shift input ids one token to the right, and wrap the last
        non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def unlikelihood_loss(self, decoder_input_ids, logits, weight_mask):
        """
        Taken from LuJunru NAPSS paper
        https://github.com/LuJunru/NapSS/blob/main/modeling/finetune.py
        decoder_input_ids - (N, s)
        logits      - (N, s, vocab_size)
        weight_mask - (N, s, vocab_size)
        """
        probs = torch.nn.functional.softmax(logits, dim=-1)
        neg_probs = 1 - probs

        # replace zeros with small positive constant for stability
        neg_probs += (neg_probs == 0).float() * 1e-8
        log_neg_probs = torch.log(neg_probs)  # (N,s,v)

        # now create attention mask and apply it
        attention_mask = decoder_input_ids.eq(1).eq(0).float()
        attention_mask = attention_mask.unsqueeze(2).expand(-1, -1, logits.shape[2])
        log_neg_probs_masked = log_neg_probs * attention_mask

        # apply weight vector to the log probability tensor
        weighted_probs = log_neg_probs_masked * weight_mask

        # TODO: take into account batch size (doesn't matter now since N=1)
        return -torch.sum(weighted_probs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models
        return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if args.loss_type[:2] == "ul" and self.is_in_train:
            # Get the logits, labels, and matrix sizes
            logits = outputs["logits"]
            labels = inputs["labels"]
            # labels = self.shift_tokens_right(labels, self.tokenizer.pad_token_id)
            batch_size, seq_len, vocab_size = logits.size()
            
            # Generate UL weights
            ul_weight = torch.zeros((batch_size, 
                                     seq_len, 
                                     vocab_size)).float().cuda()

            selective_penalty   = True
            readability_penalty = True

            # Selectivity Option
            if selective_penalty:
                # Generate logits indices mask to determine generated words
                # This is called selective penalty in the NAPSS paper
                logits_indices = torch.argmax(logits, dim=-1)
                logits_indices_mask = torch.nn.functional.one_hot(
                    logits_indices, num_classes=vocab_size
                )  # (N,s,v)
            else:
                logits_indices_mask = torch.ones(
                    batch_size, seq_len, vocab_size
                )

            # Readability Penalty
            if readability_penalty:
                read_mask = self.ul_weights
                read_mask = (
                    read_mask.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, seq_len, vocab_size)
                    .clone()
                )
                # Applying this penalizes ONLY the generated words by their complexity
                # This improves readability
                ul_weight += lambda_read * read_mask * logits_indices_mask

            # Hallucination Penalty
            hallucination_penalty = []
            if "_inp" in args.loss_type:
                hallucination_penalty.append("inputs")
            if "_lab" in args.loss_type:
                hallucination_penalty.append("labels")

            if hallucination_penalty:
                hall_mask = (
                    torch.zeros((batch_size, seq_len, vocab_size)).float().cuda()
                )
                if "labels" in hallucination_penalty:
                    # This is a penalty on words which are not in the labels
                    # This reduces hallucination
                    labels_indices_mask = torch.nn.functional.one_hot(
                        labels, num_classes=vocab_size
                    )
                    hall_mask += (
                        labels_indices_mask.sum(axis=1)
                        .unsqueeze(1)
                        .expand(batch_size, seq_len, vocab_size)
                    )
                if "inputs" in hallucination_penalty:
                    inputs_indices_mask = torch.nn.functional.one_hot(
                        inputs["input_ids"], num_classes=vocab_size
                    )
                    hall_mask += (
                        inputs_indices_mask.sum(axis=1)
                        .unsqueeze(1)
                        .expand(batch_size, seq_len, vocab_size)
                    )
                # Get the tokens which do not appear in either label or input
                neg_indices_mask = 1.0 * (hall_mask == 0) 
                # Penalize these non-appearing tokens with a fixed weight
                ul_weight += lambda_const * neg_indices_mask * logits_indices_mask

            ul_loss = self.unlikelihood_loss(
                decoder_input_ids=labels, logits=logits, weight_mask=ul_weight
            )
            loss = outputs["loss"]
            loss += ul_loss

        elif labels is not None:
            if (
                unwrap_model(model)._get_name()
                in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
            ):
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, "
                    "only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs"
                    "it received are "
                    f"{','.join(inputs.keys())}."
                )
            else:
                # We don't use .loss here since the model may return tuples
                # instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def decode_and_compute_metrics(eval_pred):
    pred_raw, label_raw, source_raw = eval_pred

    if isinstance(pred_raw, tuple):
        pred_raw = pred_raw[0]
        print("preds again", pred_raw)

    pred_raw[pred_raw == -100]     = 0
    source_raw[source_raw == -100] = 0
    label_raw[label_raw == -100]   = 0

    predictions = tokenizer.batch_decode(pred_raw)
    sources = tokenizer.batch_decode(source_raw)
    labels = tokenizer.batch_decode(label_raw)
    labels = [[s] for s in labels]  # labels must be a list of LISTS

    # with open("with_context.pkl", "wb") as f:  # open a text file
    #     pickle.dump([sources, predictions, labels], f)  # serialize the list

    result = compute_metrics(
        sources, predictions, labels, ["fkgl_easse", "ari_score", "sari_easse"]
    )
    return result


def model_init_func(trial):
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name_dict[args.model][1] if args.checkpoint == None else args.checkpoint
    )


def encode(examples):
    """This function takes a batch of samples,
    and tokenizes them into IDs for the model."""
    # Tokenize the Findings (the input)
    input_str = examples["input"]
    model_inputs = tokenizer(
        input_str, max_length=768, padding=True, truncation=True, return_tensors="pt"
    )

    # Tokenize the Impressions (the output)
    labels = tokenizer(
        [lst[0] for lst in examples["labels"]],
        max_length=768,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Set the label as the token ids (i.e. the vocab IDs) of the findings
    model_inputs["labels"] = labels["input_ids"]

    # Add anything else here and return it as an additional output
    # HERE!

    return model_inputs


def train(config=None, project=None):

    with wandb.init(config=config, project=project):

        config = wandb.config
        print(f"On Run ID: {wandb.run.id}")
        print(f"Using: {config}")

        EFFECTIVE_BATCH = int(config.batch_size) * int(
            config.gradient_accumulation_steps
        )

        training_args = Seq2SeqTrainingArguments(
            f"models/{MODEL_OUT_NAME}/{wandb.run.id}",
            # Training parameters
            num_train_epochs=int(config.epochs),
            learning_rate=float(config.learning_rate),
            lr_scheduler_type="constant",
            warmup_steps=int(config.warmup_steps),
            per_device_train_batch_size=int(config.batch_size),
            gradient_accumulation_steps=int(config.gradient_accumulation_steps),
            weight_decay=float(config.weight_decay),
            fp16=False,
            # Evaluation parameters
            evaluation_strategy="steps",  # "epoch",
            eval_steps=500, 
            # metric_for_best_model="sari",
            per_device_eval_batch_size=2,  # int(config.batch_size),
            predict_with_generate=True,
            generation_max_length=768,
            include_inputs_for_metrics=True,
            # Logging parameters
            logging_strategy="steps",
            logging_steps=1,
            run_name=f"{DATASET_NAME}_{MODEL_NAME}_{EFFECTIVE_BATCH}_{config.learning_rate}",
            report_to="wandb",
            # Saving parameters
            save_strategy="epoch", # "steps"
            # load_best_model_at_end=True,
            save_total_limit=5,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer)

        # Create the Trainer and train
        trainer = SimplificationTrainer(
            # model = model if not args.hyperparameter_tune else None,
            model_init=model_init_func,  # if args.hyperparameter_tune else None,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=decode_and_compute_metrics,
            # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )

        if args.loss_type[:2] == "ul":
            with open("fk_weights.pkl", "rb") as f:
                ul_weights = pickle.load(f)
            ul_weights = list(map(lambda x: max(x, 0.0), ul_weights))
            trainer.ul_weights = torch.tensor(ul_weights).float().cuda()

        if args.predict_only != "True":
            trainer.train()

        if args.hyperparameter_tune != "True":
            # Use the model to generate outputs
            test_output = trainer.predict(dataset["test"])
            test_output = tokenizer.batch_decode(test_output.predictions)
            test_output = list(
                map(
                    lambda s: s.replace("<s>", "")
                    .replace("</s>", "")
                    .replace("<pad>", ""),
                    test_output,
                )
            )
            return test_output
        else:
            return None


sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {"value": 1},
        "gradient_accumulation_steps": {"values": [1, 2, 4]},
        "epochs": {"value": 1},
        "learning_rate": {"distribution": "uniform", "max": 1e-4, "min": 1e-6},
        "weight_decay": {"distribution": "uniform", "max": 0.05, "min": 0.0},
        "warmup_steps": {"value": 0},
    },
}

# Turn off WANDB if predicting only
if args.predict_only == "True":
    os.system("wandb offline")
else:
    os.system("wandb online")

# Tokenizer
if args.predict_only == "True":
    tokenizer = AutoTokenizer.from_pretrained(model_name_dict[args.model][1])
else:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_dict[args.model][1] if args.checkpoint == None else args.checkpoint
    )

# Naming variables
DATASET_NAME = args.dataset
MODEL_NAME = model_name_dict[args.model][0]
PRETRAIN_NAME = (
    ""
    if (args.checkpoint is None) or ("PRETRAIN" not in args.checkpoint)
    else "_PRETRAIN"
)
if args.predict_only == "True":
    if "_ul_2voc" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_2voc"
    elif "_ul_inp_lab" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_inp_lab"
    elif "_ul_inp" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_inp"
    elif "_ul_lab" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_lab"
    elif "_ul_sel" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_sel"
    elif "_ul" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul"
    else:
        LOSS_TYPE_NAME = ""
else:
    LOSS_TYPE_NAME = "" if args.loss_type == "standard" else f"_{args.loss_type}"
MODEL_OUT_NAME = f"{MODEL_NAME}{PRETRAIN_NAME}_{DATASET_NAME}{LOSS_TYPE_NAME}{args.suffix}"
PROJECT_NAME = f"{DATASET_NAME}_{args.model}{PRETRAIN_NAME.lower()}{LOSS_TYPE_NAME}"

# Load and preprocess the datasets
dataset = load_dataset("json", data_files=f"data/{DATASET_NAME}.json", field="train")
dataset["test"] = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="test"
)["train"]


def split_sent(s):
    return sent_tokenize(s)

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

# We apply the function to all the examples in our train and test datasets
dataset["train"] = dataset["train"].map(encode, batched=True)
dataset["test"] = dataset["test"].map(encode, batched=True)

# Remove the original columns
dataset["train"] = dataset["train"].remove_columns(["input"])
dataset["test"] = dataset["test"].remove_columns(["input"])

try:
    dataset["train"] = dataset["train"].remove_columns(["vocab", "report_id"])
    dataset["test"] = dataset["test"].remove_columns(["vocab", "report_id"])
except:
    pass

if args.hyperparameter_tune == "True":
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train, count=args.hyperparameter_trials)
else:
    config = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
    }
    test_output = train(config, PROJECT_NAME)

    # Collate sentences by report
    if args.pred_split_sent == "True":
        df["prediction"] = test_output
        df = (
            df.groupby("report_id")
            .aggregate({"prediction": lambda lst: ". ".join(lst)})
            .reset_index(drop=True)
        )
        test_output = list(df["prediction"])

    # Write output
    with open(f"output/{PROJECT_NAME}{args.suffix}.txt", "w") as fp:
        for item in test_output:
            fp.write("%s\n" % item)
        print("Done")
