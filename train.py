import pandas as pd
import numpy as np
import nltk
from datasets import Dataset, DatasetDict, load_metric, load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import argparse

metric_rouge = load("rouge")
metric_bertscore = load("bertscore")
metric_sari = load("sari")


def compute_metrics(eval_pred):
    predictions, labels, sources = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]
        print("preds again", predictions)

    # Replace -100 in the labels and sources as we can't decode them.
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    sources = np.where(sources != -100, sources, tokenizer.pad_token_id)
    decoded_inputs = tokenizer.batch_decode(sources, skip_special_tokens=True)

    # Tokenize and clean
    decoded_preds_newln = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_preds_space = [
        " ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_label_newln = [
        "\n".join(nltk.sent_tokenize(labl.strip())) for labl in decoded_labels
    ]
    decoded_label_space = [
        " ".join(nltk.sent_tokenize(labl.strip())) for labl in decoded_labels
    ]
    decoded_input_space = [
        " ".join(nltk.sent_tokenize(inpt.strip())) for inpt in decoded_inputs
    ]

    result_rouge = metric_rouge.compute(
        predictions=decoded_preds_newln,
        references=decoded_label_newln,
        use_stemmer=True,
    )
    result_berts = metric_bertscore.compute(
        predictions=decoded_preds_space, references=decoded_label_space, lang="en"
    )
    result_sari = metric_sari.compute(
        sources=decoded_input_space,
        predictions=decoded_preds_space,
        references=[[i] for i in decoded_label_space],
    )

    # Extract results
    result = result_rouge  # {key: value.mid.fmeasure * 100 for key, value in result_rouge.items()}
    result["bert_score"] = np.mean(result_berts["f1"])
    result["sari"] = result_sari["sari"]
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# Get dataset from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--predict_only", required=False, type=bool, default=False)
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
print(
    f"Using dataset: {args.dataset}, Args: {args.lr} (lr), {args.epochs} (epochs), {args.batch_size} (batch_size)"
)
print(
    f"{args.gradient_accumulation_steps} (gradient_accumulation_steps), {args.weight_decay} (weight_decay), {args.checkpoint} (checkpoint)"
)

DATASET_NAME = args.dataset
dataset = load_dataset("json", data_files=f"data/{DATASET_NAME}.json", field="train")
dataset["test"] = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="test"
)["train"]

# Load in the model and tokenizer, for this we're using BART, which is good at generation tasks
if args.model == "bart":
    MODEL_NAME = "BART"
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large" if args.checkpoint == None else args.checkpoint
    )
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
elif args.model == "flant5":
    MODEL_NAME = "FLANT5_LARGE"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large" if args.checkpoint == None else args.checkpoint
    )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
elif args.model == "flant5_base":
    MODEL_NAME = "FLANT5_BASE"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base" if args.checkpoint == None else args.checkpoint
    )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
else:
    assert False


def preprocess_function(examples):
    """This function takes a batch of samples, and tokenizes them into IDs for the model
       It does this by adding new arguments to the Dataset dictionary, namely
       - input_ids:      tokenized IDs of the findings
       - attention_mask: mask that tells us which tokens are words and which are padding
       - labels:         tokenized IDs of the impressions
    Args:
        examples (Dataset): {'Findings':[<list of findings texts>],
                             'Impressions':[[<list of impressions texts>] per item]}

    Returns:
        model_inputs (Dataset): {'Findings':      [<list of findings texts>],
                                 'Impressions':   [<list of impressions texts>],
                                 'input_ids':     list of lists with impressions IDs,
                                 'attention_mask':list of lists with impressions IDs masks,
                                 'labels':        list of lists with findings IDs}
    """
    # Tokenize the Findings (the input)
    # input_str = list(map(lambda s: f"Simplify: {s}", examples["input"])) if args.model == 'flant5' else examples["input"]
    input_str = examples["input"]
    model_inputs = tokenizer(input_str, max_length=768, padding=True, truncation=True)
    # Tokenize the Impressions (the output)
    labels = tokenizer(
        [lst[0] for lst in examples["labels"]],
        max_length=768,
        padding=True,
        truncation=True,
    )
    # Set the label as the token ids (i.e. the vocab IDs) of the findings
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# We apply the function to all the examples in our train and test datasets
dataset["train"] = dataset["train"].map(preprocess_function, batched=True)
dataset["test"] = dataset["test"].map(preprocess_function, batched=True)

# Remove the original columns
dataset["train"] = dataset["train"].remove_columns(["input"])
dataset["test"] = dataset["test"].remove_columns(["input"])

try:
    dataset["train"] = dataset["train"].remove_columns(["vocab", "report_id"])
    dataset["test"] = dataset["test"].remove_columns(["vocab", "report_id"])
except:
    pass

# Write out the arguments
MODEL_OUT_NAME = f"{MODEL_NAME}_{DATASET_NAME}"

EFFECTIVE_BATCH = int(args.batch_size) * int(args.gradient_accumulation_steps)

training_args = Seq2SeqTrainingArguments(
    f"models/{MODEL_OUT_NAME}",
    # Training parameters
    num_train_epochs=int(args.epochs),
    learning_rate=float(args.lr),
    warmup_steps=int(args.warmup_steps),
    per_device_train_batch_size=int(args.batch_size),
    gradient_accumulation_steps=int(args.gradient_accumulation_steps),
    weight_decay=float(args.weight_decay),
    fp16=False,
    # Evaluation parameters
    evaluation_strategy="epoch",
    per_device_eval_batch_size=int(args.batch_size),
    predict_with_generate=True,
    generation_max_length=100,
    include_inputs_for_metrics=True,
    # Logging parameters
    logging_strategy="steps",
    logging_steps=1,
    run_name=f"{args.dataset}_{MODEL_NAME}_{EFFECTIVE_BATCH}_{args.lr}",
    report_to="wandb",
    # Saving parameters
    save_strategy="epoch",
    load_best_model_at_end=True,
    # save_steps = 100 if 'radiology' in args.dataset else 1000,
    # save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

# Create the Trainer and train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if not args.predict_only:
    trainer.train()

# Use the model to generate outputs
test_output = trainer.predict(dataset["test"])
test_output = tokenizer.batch_decode(test_output.predictions)
test_output = list(
    map(
        lambda s: s.replace("<s>", "").replace("</s>", "").replace("<pad>", ""),
        test_output,
    )
)

# open file in write mode
with open(f"output/{args.dataset}_{args.model}.txt", "w") as fp:
    for item in test_output:
        fp.write("%s\n" % item)
    print("Done")
