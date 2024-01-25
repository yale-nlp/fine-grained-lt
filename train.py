import argparse
import loss_library
import os
import pandas as pd
import torch
import wandb

from datasets import load_dataset, Dataset
from nltk.tokenize import sent_tokenize
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from typing import Any, Dict, List, Optional, Tuple, Union
from utils.utils_eval import compute_metrics

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
        
        outputs = model(**inputs)

        if self.is_in_train and self.loss_function is not None:
            
            # Retrieve logits and labels
            logits = outputs["logits"]
            labels = inputs["labels"].to(logits.device)
            
            loss = self.loss_function(logits = logits, labels = labels, inputs = inputs)

        else:
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
            save_strategy="epoch",
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
        )

        # Instantiate a loss type from LossLibrary 
        # and add loss specific arguments to the trainer

        # Unlikelihood Loss to penalize complex and unsupported words
        if args.loss_type == "ul":
            trainer.loss_function = loss_library.LossLibrary(
                loss_type = "ul", 
                tokenizer = trainer.tokenizer, 
                model = trainer.model, 
                ul_weights_path = f"{ROOT_PATH}/assets/fk_weights.pkl",
                ul_lambda_read  = 5e-4,
                ul_lambda_const = 1.5e-4,
                ul_check_input_ents = True,
                ul_check_label_ents = True
                )

        # Rejection Loss to penalize model uncertainly and upweight 
        # probability of an UNK token instead
        elif args.loss_type == "rej":
            trainer.loss_function = loss_library.LossLibrary(
                loss_type = "rej", 
                tokenizer = trainer.tokenizer, 
                model = trainer.model
            )
        
        # Loss Truncation
        elif args.loss_type in ["lt", "max_lt", "mi_lt"]:
            trainer.loss_function = loss_library.LossLibrary(
                loss_type = args.loss_type,
                tokenizer = trainer.tokenizer, 
                model = trainer.model,
                lt_dropc = 0.4,
                lt_min_count = 500,
                lt_recompute = 500
            )

        # Mutual Information Augmented Loss
        elif args.loss_type == "mi":
            trainer.loss_function = loss_library.LossLibrary(
                loss_type = args.loss_type,
                tokenizer = trainer.tokenizer, 
                model = trainer.model,
                mi_weight = 1.0,
                mi_filter = "all"
            )
        # If none of the special loss types are specified, use the standard
        # NLL loss
        else:
            trainer.loss_function = None

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

# Format the output names with the specific loss function and dataset configurations
if args.predict_only == "True":

    # If we're predicting from a checkpoint, find the loss with which the model checkpoint
    # was trained, and add that to the output name
    loss_keywords = ["_cutoff", "_ul_2voc", "_ul_inp_lab", "_ul_inp", "_ul_lab", "_ul_sel", "_ul_ent", "_ul_lt", "_mi_ul", "_ul", "_rej", "_lt", "_max_lt", "_mi_lt", "_mi", "_copy"]
    
    if args.checkpoint is None: LOSS_TYPE_NAME = ""
    else: LOSS_TYPE_NAME = "".join([s for s in loss_keywords if s in args.checkpoint])
    
    # If we're predicting from a checkpoint, find any specific properties of the dataset used
    # to train the model, and add that to the output name
    dataset_keywords = ["_drop_unk", "_aug_ents"]
    if args.checkpoint is None: DATASET_PROPERTY = ""
    else: DATASET_PROPERTY = "".join([s for s in dataset_keywords if s in args.checkpoint])

    # If we're doing a prediction on the train set, add that to the output name
    if args.predict_train != "False":
        DATASET_PROPERTY += "_trainset"
else:
    LOSS_TYPE_NAME = "" if args.loss_type == "standard" else f"_{args.loss_type}"
    DATASET_PROPERTY = ""

MODEL_OUT_NAME = f"{MODEL_NAME}{PRETRAIN_NAME}_{DATASET_NAME}{DATASET_PROPERTY}{LOSS_TYPE_NAME}{args.suffix}"
PROJECT_NAME = f"{DATASET_NAME}{DATASET_PROPERTY}_{args.model}{PRETRAIN_NAME.lower()}{LOSS_TYPE_NAME}"

# Load the datasets
ROOT_PATH = "simplification-project"

dataset = load_dataset(
    "json", 
    data_files=f"{ROOT_PATH}/data/{DATASET_NAME}.json", 
    field="train")
dataset["test"] = load_dataset(
    "json", 
    data_files=f"{ROOT_PATH}/data/{DATASET_NAME}_multiple.json", 
    field="test"
)["train"]

# Tokenize the datasets
columns_to_remove = list(set(dataset["train"].columns).difference(set(["input_ids", "attention_mask", "labels"])))
dataset["train"] = dataset["train"].map(encode, batched=True, remove_columns=columns_to_remove)
dataset["test"] = dataset["test"].map(encode, batched=True, remove_columns=columns_to_remove)

# If hyperparameter tuning is on, run training using wandb.sweep
if args.hyperparameter_tune == "True":
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train, count=args.hyperparameter_trials)
    
# Otherwise, run one train loop with the given configuration
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

    # Write output
    with open(f"output/{PROJECT_NAME}{args.suffix}.txt", "w") as fp:
        for item in test_output:
            fp.write("%s\n" % item)
        print("Done")
