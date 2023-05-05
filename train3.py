import argparse
import wandb
import numpy as np

from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from utils_eval import compute_metrics
from utils_eval import compute_metrics, calculate_sari, clean_string
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

# Get dataset from arguments
parser = argparse.ArgumentParser()

# Dataset and model
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--checkpoint", required=False, type=str, default=None)

# Script mode
parser.add_argument("--hyperparameter_tune", required=False, type=str, default="False")
parser.add_argument("--hyperparameter_trials", required=False, type=int, default=5)
parser.add_argument("--predict_only", required=False, type=str, default="False")

# Training parameters
parser.add_argument("--learning_rate", required=False, type=float, default=1e-5)
parser.add_argument("--epochs", required=False, type=int, default=5)
parser.add_argument("--batch_size", required=False, type=int, default=1)
parser.add_argument("--weight_decay", required=False, type=float, default=0.01)
parser.add_argument("--loss_type", required=False, type=str, default="standard")
parser.add_argument(
    "--gradient_accumulation_steps", required=False, type=int, default=1
)
parser.add_argument("--warmup_steps", required=False, type=int, default=1000)
args = parser.parse_args()

assert args.hyperparameter_tune in ["True", "False"]
assert args.predict_only in ["True", "False"]
assert args.loss_type in ["standard", "rl", "rl_policy"]

# Load in the model and tokenizer, for this we're using BART, which is good at generation tasks
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
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if args.loss_type in ["rl", "rl_policy"] and self.is_in_train:

            # Get and clean the orig prediction, source, and reference
            decoded_output = self.tokenizer.batch_decode(
                outputs["logits"].argmax(dim=2)
            )
            decoded_input = self.tokenizer.batch_decode(inputs["input_ids"])
            decoded_labels = self.tokenizer.batch_decode(inputs["labels"])

            decoded_output = list(map(clean_string, decoded_output))
            decoded_input = list(map(clean_string, decoded_input))
            decoded_labels = list(map(clean_string, decoded_labels))

            decoded_labels = [[s] for s in decoded_labels]

            # Compute SARI
            sari_scores = [
                calculate_sari([s], [p], [l])["sari"]
                for (s, p, l) in zip(decoded_input, decoded_output, decoded_labels)
            ]

            # Compute cross entropy loss
            criterion = torch.nn.CrossEntropyLoss(reduction="none")
            logits = outputs["logits"]
            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(
                logits.view(-1, logits.shape[-1]), inputs["labels"].view(-1)
            )
            loss = loss.view(batch_size, seq_len).mean(axis=1)
            # loss = loss / loss.item() # TODO: Check if this should be removed, discuss

            if args.loss_type == "rl":
                # Rescale SARI for numeric stability
                sari_scores = torch.tensor(
                    [0.01 * (100.0 - s) for s in sari_scores]
                ).cuda()

                # Scale cross entropy loss by the SARI score
                loss = torch.sum(loss * sari_scores)

            if args.loss_type == "rl_policy":
                # Run a prediction step using sampling decoding
                inputs_copy = inputs.copy()
                if "labels" in inputs_copy:
                    inputs_copy.pop("labels")

                outputs_sampling = self.model.generate(
                    **inputs_copy,
                    do_sample=True,
                    top_k=0,
                )

                # Compute SARI of sampling decoding labels
                decoded_output_sampling = self.tokenizer.batch_decode(outputs_sampling)
                decoded_output_sampling = list(
                    map(clean_string, decoded_output_sampling)
                )
                sari_scores_sampling = [
                    calculate_sari([s], [p], [l])["sari"]
                    for (s, p, l) in zip(
                        decoded_input, decoded_output_sampling, decoded_labels
                    )
                ]

                # Subtract sari_sampling - sari_base, rescale for stability
                sari_scores = torch.tensor(sari_scores) - torch.tensor(
                    sari_scores_sampling
                )
                sari_scores = 0.01 * sari_scores.cuda()

                # Scale cross entropy loss by the SARI score
                loss = torch.sum(loss * sari_scores)

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
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are "
                    f"{','.join(inputs.keys())}."
                )
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def decode_and_compute_metrics(eval_pred):
    pred_raw, label_raw, source_raw = eval_pred

    if isinstance(pred_raw, tuple):
        pred_raw = pred_raw[0]
        print("preds again", pred_raw)

    predictions = tokenizer.batch_decode(pred_raw)
    sources = tokenizer.batch_decode(source_raw)
    labels = tokenizer.batch_decode(label_raw)
    labels = [[s] for s in labels]  # labels must be a list of LISTS

    return compute_metrics(sources, predictions, labels, ["rouge", "sari"])


def model_init_func(trial):
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name_dict[args.model][1] if args.checkpoint == None else args.checkpoint
    )


def encode(examples):
    """This function takes a batch of samples, and tokenizes them into IDs for the model."""
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
            warmup_steps=int(config.warmup_steps),
            per_device_train_batch_size=int(config.batch_size),
            gradient_accumulation_steps=int(config.gradient_accumulation_steps),
            weight_decay=float(config.weight_decay),
            fp16=False,
            # Evaluation parameters
            evaluation_strategy="epoch",
            per_device_eval_batch_size=1,  # int(config.batch_size),
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
            load_best_model_at_end=True,
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

            # open file in write mode
            with open(f"output/{PROJECT_NAME}.txt", "w") as fp:
                for item in test_output:
                    fp.write("%s\n" % item)
                print("Done")


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

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_dict[args.model][1])

# Naming variables
DATASET_NAME = args.dataset
MODEL_NAME = model_name_dict[args.model][0]
PRETRAIN_NAME = (
    ""
    if (args.checkpoint is None) or ("PRETRAIN" not in args.checkpoint)
    else "_PRETRAIN"
)
LOSS_TYPE_NAME = "" if args.loss_type == "standard" else f"_{args.loss_type}"
MODEL_OUT_NAME = f"{MODEL_NAME}{PRETRAIN_NAME}_{DATASET_NAME}{LOSS_TYPE_NAME}"
PROJECT_NAME = f"{DATASET_NAME}_{args.model}{PRETRAIN_NAME.lower()}{LOSS_TYPE_NAME}"

# Load and preprocess the datasets
dataset = load_dataset("json", data_files=f"data/{DATASET_NAME}.json", field="train")
dataset["test"] = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="test"
)["train"]

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
    train(config, PROJECT_NAME)
