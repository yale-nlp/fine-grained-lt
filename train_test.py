import argparse
import math
import loss_library
import os
import pandas as pd
import pickle
import spacy
import scispacy
import torch
import torch.nn.functional as F
import wandb

from datasets import load_dataset, Dataset
from nltk.tokenize import sent_tokenize
from scispacy.linking import EntityLinker
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
from torch.nn import NLLLoss, CrossEntropyLoss
from typing import Any, Dict, List, Optional, Tuple, Union
# from utils_loss import (rejection_loss, unlikelihood_loss, loss_truncation, mutual_information_loss)
from utils_loss_truncate import LossDropper
from utils_eval import compute_metrics

torch.autograd.set_detect_anomaly(True)

# Get dataset from arguments
parser = argparse.ArgumentParser()

# GPU local rank
parser.add_argument("--local_rank", required=False, type=int, default=None)

# Dataset and model
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--checkpoint", required=False, type=str, default=None)

# Naming
parser.add_argument("--suffix", required=False, type=str, default="")

# Hyperparameter tuning mode
parser.add_argument("--hyperparameter_tune", required=False, type=str, default="False")
parser.add_argument("--hyperparameter_trials", required=False, type=int, default=5)

# Predict mode
parser.add_argument("--predict_train", required=False, type=str, default="False")
parser.add_argument("--predict_only", required=False, type=str, default="False")
parser.add_argument("--pred_split_sent", required=False, type=str, default="False")

# Loss type
parser.add_argument("--loss_type", required=False, type=str, default="standard")
parser.add_argument("--loss_cutoff", required=False, type=int, default=0)

# Training parameters
parser.add_argument("--learning_rate", required=False, type=float, default=5e-5)
parser.add_argument("--epochs", required=False, type=int, default=1)
parser.add_argument("--batch_size", required=False, type=int, default=1)
parser.add_argument("--weight_decay", required=False, type=float, default=0.01)
parser.add_argument("--adam_epsilon", required=False, type=float, default=1e-8)
parser.add_argument("--max_grad_norm", required=False, type=float, default=1.00)
parser.add_argument(
    "--gradient_accumulation_steps", required=False, type=int, default=1
)
parser.add_argument("--warmup_steps", required=False, type=int, default=0)
parser.add_argument("--scheduler", required=False, type=str, default="constant")
args = parser.parse_args()

if args.local_rank is not None:
    torch.cuda.set_device(args.local_rank)

assert args.hyperparameter_tune in ["True", "False"]
assert args.predict_only in ["True", "False"]

# Load in the model and tokenizer, for this we're using BART,
# which is good at generation tasks
model_name_dict = {
    "bart": ("BART", "facebook/bart-large"),
    "bart_xsum": ("BART_XSUM", "facebook/bart-large-xsum"),
    "flant5": ("FLANT5_LARGE", "google/flan-t5-large"),
    "flant5_base": ("FLANT5_BASE", "google/flan-t5-base"),
}

class SimplificationTrainer(Seq2SeqTrainer):

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
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.is_in_train:
            
            # Retrieve logits and labels
            logits = outputs["logits"]
            labels = inputs["labels"].to(logits.device)
            
            loss = self.loss_function(logits = logits, labels = labels, inputs = inputs)

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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_dict[args.model][1] if args.checkpoint is None else args.checkpoint
    )
    if args.local_rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
            )
    return model


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
    if "meets_cutoff" in examples:
        model_inputs["meets_cutoff"] = [1 if x=="True" else 0 for x in examples['meets_cutoff']]
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
            lr_scheduler_type=config.scheduler,
            adam_epsilon=float(args.adam_epsilon),
            max_grad_norm=float(config.max_grad_norm),
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

        # Add loss specific arguments to the trainer
        if args.loss_type == "ul":
            trainer.loss_function = loss_library.LossLibrary(
                loss_type = "ul", 
                tokenizer = trainer.tokenizer, 
                model = trainer.model, 
                ul_weights_path = f"{ROOT_PATH}/fk_weights.pkl",
                ul_lambda_read  = 5e-4,
                ul_lambda_const = 1.5e-4,
                ul_check_input_ents = True,
                ul_check_label_ents = True
                )

            
        # Rejection Loss
        if args.loss_type == "rej":
            trainer.loss_function = loss_library.LossLibrary(
                loss_type = "rej", 
                tokenizer = trainer.tokenizer, 
                model = trainer.model
            )
        
        if args.loss_type in ["lt", "max_lt", "mi_lt"]:
            trainer.loss_function = loss_library.LossLibrary(
                loss_type = args.loss_type,
                tokenizer = trainer.tokenizer, 
                model = trainer.model,
                lt_dropc = 0.4,
                lt_min_count = 500,
                lt_recompute = 500
            )

        if args.loss_type == "mi":
            trainer.loss_function = loss_library.LossLibrary(
                loss_type = args.loss_type,
                tokenizer = trainer.tokenizer, 
                model = trainer.model,
                mi_weight = 1.0,
                mi_filter = "all"
            )

        # Train the model
        if args.predict_only != "True":
            trainer.train()

        # Do prediction
        if args.hyperparameter_tune != "True":
            # Use the model to generate outputs
            test_output = trainer.predict(dataset["test"]) if args.predict_train == "False" else trainer.predict(dataset["train"])
            test_output = tokenizer.batch_decode(test_output.predictions)
            test_output = list(
                map(
                    lambda s: s.replace("<s>", "")
                    .replace("</s>", "")
                    .replace("<pad>", "")
                    .replace("\n", ""),
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
        model_name_dict[args.model][1] if args.checkpoint is None else args.checkpoint
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
    if args.checkpoint is None:
        LOSS_TYPE_NAME = ""
    elif "_cutoff" in args.checkpoint:
        LOSS_TYPE_NAME = "_cutoff"
    elif "_ul_2voc" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_2voc"
    elif "_ul_inp_lab" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_inp_lab"
    elif "_ul_inp" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_inp"
    elif "_ul_lab" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_lab"
    elif "_ul_sel" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_sel"
    elif "_ul_ent" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_ent"
    elif "_ul_lt" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul_lt"
    elif "_mi_ul" in args.checkpoint:
        LOSS_TYPE_NAME = "_mi_ul"
    elif "_ul" in args.checkpoint:
        LOSS_TYPE_NAME = "_ul"
    elif "_rej" in args.checkpoint:
        LOSS_TYPE_NAME = "_rej"
    elif "_lt" in args.checkpoint:
        LOSS_TYPE_NAME = "_lt"
    elif "_max_lt" in args.checkpoint:
        LOSS_TYPE_NAME = "_max_lt"
    elif "_mi_lt" in args.checkpoint:
        LOSS_TYPE_NAME = "_mi_lt"
    elif "_mi" in args.checkpoint:
        LOSS_TYPE_NAME = "_mi"
    elif "_copy" in args.checkpoint:
        LOSS_TYPE_NAME = "_copy"
    else:
        LOSS_TYPE_NAME = ""

    if args.checkpoint is None:
        DATASET_PROPERTY = ""
    elif "_drop_unk" in args.checkpoint:
        DATASET_PROPERTY = "_drop_unk"
    elif "_aug_ents" in args.checkpoint:
        DATASET_PROPERTY = "_aug_ents"
    else:
        DATASET_PROPERTY = ""

    if args.predict_train != "False":
        DATASET_PROPERTY += "_trainset"
else:
    LOSS_TYPE_NAME = "" if args.loss_type == "standard" else f"_{args.loss_type}"
    DATASET_PROPERTY = ""

MODEL_OUT_NAME = f"{MODEL_NAME}{PRETRAIN_NAME}_{DATASET_NAME}{DATASET_PROPERTY}{LOSS_TYPE_NAME}{args.suffix}"
PROJECT_NAME = f"{DATASET_NAME}{DATASET_PROPERTY}_{args.model}{PRETRAIN_NAME.lower()}{LOSS_TYPE_NAME}"

# Load and preprocess the datasets
ROOT_PATH = "/home/lyf6/simplification-project"
dataset = load_dataset(
    "json", 
    data_files=f"{ROOT_PATH}/data/{DATASET_NAME}.json", 
    field="train")
dataset["test"] = load_dataset(
    "json", 
    data_files=f"{ROOT_PATH}/data/{DATASET_NAME}_multiple.json", 
    field="test"
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
        "scheduler": args.scheduler,
        "max_grad_norm": args.max_grad_norm
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
