import argparse
import itertools
import numpy as np
import pandas as pd
import pickle
import spacy
import torch
import wandb

from datasets import load_dataset, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from utils_eval import (
    compute_metrics,
    calculate_sari,
    calculate_questeval,
    clean_string,
    get_readability_score,
)
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Load in the model and tokenizer, for this we're using BART, which is good at generation tasks
model_name_dict = {
    "bart": ("BART", "facebook/bart-large"),
    "bart_xsum": ("BART_XSUM", "facebook/bart-large-xsum"),
    "flant5": ("FLANT5_LARGE", "google/flan-t5-large"),
    "flant5_base": ("FLANT5_BASE", "google/flan-t5-base"),
}

if args.loss_type in ["rl_qe", "rl_policy_qe"]:
    from questeval.questeval_metric import QuestEval

    questeval = QuestEval(no_cuda=False)


class SimplificationTrainer(Seq2SeqTrainer):
    def shift_tokens_right(self, input_ids, pad_token_id):
        """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def unlikelihood_loss(
        self, decoder_input_ids, logits, weight_mask, selective_penalty
    ):
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
        # N,s = logits.size()[:2]
        # weight_mask = weight_mask.unsqueeze(0).unsqueeze(0).expand(N,s,-1)
        weighted_probs = log_neg_probs_masked * weight_mask

        if selective_penalty:
            indices = torch.argmax(logits, dim=-1)
            indices_mask = torch.nn.functional.one_hot(
                indices, num_classes=logits.shape[-1]
            )  # (N,s,v)
            weighted_probs *= indices_mask

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

        # # Get and clean the predictions, and orig source and reference
        decoded_input = inputs["input_ids"].detach().clone()
        decoded_input[decoded_input == -100] = 0

        decoded_output = outputs["logits"].detach().clone().argmax(dim=2)
        print(decoded_output)

        decoded_labels = inputs["labels"].detach().clone()
        decoded_labels[decoded_labels == -100] = 0

        decoded_input = self.tokenizer.batch_decode(decoded_input)
        decoded_output = self.tokenizer.batch_decode(decoded_output)
        decoded_labels = self.tokenizer.batch_decode(decoded_labels)
        print(decoded_output)

        decoded_input = list(map(clean_string, decoded_input))
        decoded_output = list(map(clean_string, decoded_output))
        decoded_labels = list(map(clean_string, decoded_labels))
        decoded_labels = [[s] for s in decoded_labels]

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if args.loss_type[:2] == "ul" and self.is_in_train:
            # Compute the weight mask
            logits = outputs["logits"]
            labels = inputs["labels"]
            labels = self.shift_tokens_right(labels, self.tokenizer.pad_token_id)

            b, s, v = logits.shape[0], logits.shape[1], logits.shape[2]
            weight_mask = (
                torch.zeros((b, v), requires_grad=True).float().cuda()
            )  # (batch_size, vocab)

            # 1. Figure out what words we want to apply UL to
            # _ent penalizes entities determined by spacy
            if "_ent" in args.loss_type:

                # Get the indices of the entities in all of the outputs
                entity_lst = [
                    list(map(str, self.ner_model(x).ents)) for x in decoded_output
                ]

                # If diff, remove the entities that are expected in the labels
                if "_entdiff" in args.loss_type:
                    entity_lst_label = [
                        list(map(str, self.ner_model(x).ents)) for x in decoded_labels
                    ]
                    entity_lst = [
                        list(set(e).difference(set(e_l)))
                        for (e, e_l) in zip(entity_lst, entity_lst_label)
                    ]

                # Get unique entities
                entity_lst = [list(set(lst)) for lst in entity_lst]

            # _all penalizes all the tokens in the output
            elif "_all" in args.loss_type:
                entity_lst = [[o] for o in decoded_output]

            else:
                assert False, print("Either need _all or _lab in the ul loss")

            # 2. Determine what weights to use for UL
            if "_fk" in args.loss_type:
                # Convert to IDs and also get the corresponding FK scores
                entity_lst = [
                    [
                        (
                            self.tokenizer(t)["input_ids"][1:-1],
                            max(
                                get_readability_score(t, metric="flesch_kincaid_grade")[
                                    0
                                ],
                                0.0,
                            ),
                        )
                        for t in set(lst)
                    ]
                    if len(lst) > 0
                    else []
                    for lst in entity_lst
                ]

                for i in range(b):
                    for (idxs, score) in entity_lst[i]:
                        weight_mask[i][idxs] = score
                    if weight_mask[i].sum() > 0.0:
                        weight_mask[i] /= weight_mask[i].sum().item()
            else:
                # Convert to IDs
                entity_lst = [
                    list(set(itertools.chain(*self.tokenizer(lst)["input_ids"][1:-1])))
                    if len(lst) > 0
                    else []
                    for lst in entity_lst
                ]

                # Set their weight uniformly (i.e. we don't want to generate them)
                for i in range(b):
                    weight_mask[i][entity_lst[i]] = 1.0
                    if weight_mask[i].sum() > 0.0:
                        weight_mask[i] /= weight_mask[i].sum().item()

            # 3. Expand it to the sequence length and apply UL
            weight_mask_expand = weight_mask.unsqueeze(1).expand(b, s, v)

            ul_loss = self.unlikelihood_loss(
                decoder_input_ids=labels,
                logits=logits,
                weight_mask=weight_mask_expand,
                selective_penalty=True,
            )
            loss = outputs["loss"]
            loss += 10 * ul_loss

        elif args.loss_type[:2] == "rl" and self.is_in_train:
            if "_fk" in args.loss_type:
                sari_scores = [
                    get_readability_score(s, metric="flesch_kincaid_grade")[0]
                    for s in decoded_output
                ]

            elif "_qe" in args.loss_type:
                sari_scores = calculate_questeval(
                    decoded_input, decoded_output, decoded_labels, questeval, both=False
                )["questeval_no_ref_raw"]
            else:
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

            if args.loss_type in ["rl", "rl_qe", "rl_fk"]:
                # Rescale SARI
                if args.loss_type in ["rl", "rl_qe"]:
                    sari_scores = [0.01 * (100.0 - s) for s in sari_scores]

                # Scale cross entropy loss by the SARI score
                sari_scores = torch.tensor(sari_scores).cuda()
                loss = torch.sum(loss * sari_scores)

            if args.loss_type in ["rl_policy", "rl_policy_qe"]:
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
                outputs_sampling[outputs_sampling == -100] = 0
                decoded_output_sampling = self.tokenizer.batch_decode(outputs_sampling)
                decoded_output_sampling = list(
                    map(clean_string, decoded_output_sampling)
                )
                if "_qe" in args.loss_type:
                    sari_scores_sampling = calculate_questeval(
                        decoded_input,
                        decoded_output_sampling,
                        decoded_labels,
                        questeval,
                        both=False,
                    )["questeval_no_ref_raw"]
                else:
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

    with open("with_context.pkl", "wb") as f:  # open a text file
        pickle.dump([sources, predictions, labels], f)  # serialize the list

    result = compute_metrics(
        sources, predictions, labels, ["rouge", "sari", "flesch_kincaid_grade"]
    )
    result.pop("flesch_kincaid_grade_counts")

    return result


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
            per_device_eval_batch_size=int(config.batch_size),
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

        if args.loss_type[:2] == "ul":
            trainer.ner_model = spacy.load("en_core_web_lg")

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

# Tokenizer
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
    if "_rl_policy_qe" in args.checkpoint:
        LOSS_TYPE_NAME = "_rl_policy_qe"
    elif "_rl_policy" in args.checkpoint:
        LOSS_TYPE_NAME = "_rl_policy"
    elif "_rl_qe" in args.checkpoint:
        LOSS_TYPE_NAME = "_rl_qe"
    elif "_rl" in args.checkpoint:
        LOSS_TYPE_NAME = "_rl"
    else:
        LOSS_TYPE_NAME = ""
else:
    LOSS_TYPE_NAME = "" if args.loss_type == "standard" else f"_{args.loss_type}"
MODEL_OUT_NAME = f"{MODEL_NAME}{PRETRAIN_NAME}_{DATASET_NAME}{LOSS_TYPE_NAME}"
PROJECT_NAME = f"{DATASET_NAME}_{args.model}{PRETRAIN_NAME.lower()}{LOSS_TYPE_NAME}"

# Load and preprocess the datasets
dataset = load_dataset("json", data_files=f"data/{DATASET_NAME}.json", field="train")
dataset["test"] = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="test"
)["train"]


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
    with open(f"output/{PROJECT_NAME}.txt", "w") as fp:
        for item in test_output:
            fp.write("%s\n" % item)
        print("Done")
