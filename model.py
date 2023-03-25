import pandas as pd
import numpy as np
import nltk
from datasets import Dataset, DatasetDict, load_metric, load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import argparse

metric_rouge = load("rouge")
metric_bertscore = load("bertscore")
metric_sari = load("sari")

# Get dataset from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()
print(f"Using dataset: {args.dataset}, model: {args.model}")

def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", 
                              "min": 1e-5 if args.model=='flant5' else 1e-6, 
                              "max": 1e-3 if args.model=='flant5' else 1e-4},
            "gradient_accumulation_steps": {"values": [4, 8, 16]},
            # "per_device_train_batch_size": {"values": [2, 4, 8]},
            "num_train_epochs": {"values": [5, 10]}
        },
    }
    
def model_init_bart(trial):
    return BartForConditionalGeneration.from_pretrained("facebook/bart-large")

def model_init_flant5(trial):
    return AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# def remove_none(lst):
#     return list(filter(lambda item: item is not None, lst))

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
    decoded_preds_newln = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_preds_space = [ " ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_label_newln = ["\n".join(nltk.sent_tokenize(labl.strip())) for labl in decoded_labels]
    decoded_label_space = [ " ".join(nltk.sent_tokenize(labl.strip())) for labl in decoded_labels]
    decoded_input_space = [ " ".join(nltk.sent_tokenize(inpt.strip())) for inpt in decoded_inputs]
    
    # sources=["About 95 species are currently accepted.","About 95 species are currently accepted."]
    # predictions=["About 95 you now get in.","About 95 you now get in."]
    # references=[["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."],
    #             ["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."]]

    result_rouge = metric_rouge.compute(predictions=decoded_preds_newln, references=decoded_label_newln, use_stemmer=True)
    result_berts = metric_bertscore.compute(predictions=decoded_preds_space, references=decoded_label_space, lang="en")
    result_sari  = metric_sari.compute(sources=decoded_input_space, predictions=decoded_preds_space, references=[[i] for i in decoded_label_space])

    # Extract results
    result = result_rouge # {key: value.mid.fmeasure * 100 for key, value in result_rouge.items()}
    result['bert_score'] = np.mean(result_berts['f1'])
    result['sari']       = result_sari['sari']
    prediction_lens      = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"]    = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

DATASET_NAME    = args.dataset 
dataset         = load_dataset('json', data_files=f'data/{DATASET_NAME}.json', field='train')
dataset['test'] = load_dataset('json', data_files=f'data/{DATASET_NAME}.json', field='test')['train']

# Load in the model and tokenizer, for this we're using BART, which is good at generation tasks
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

if args.model == 'bart':
    MODEL_NAME = "BART"
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model_init = model_init_bart

elif args.model == 'flant5':
    MODEL_NAME = 'FLANT5'
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model_init = model_init_flant5
    
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
    input_str = list(map(lambda s: f"Simplify: {s}", examples["input"])) if args.model == 'flant5' else examples["input"]
    model_inputs = tokenizer(input_str, max_length=512, padding=True, truncation=True)
    # Tokenize the Impressions (the output)
    labels = tokenizer([lst[0] for lst in examples["labels"]], max_length=512, padding=True, truncation=True)
    # Set the label as the token ids (i.e. the vocab IDs) of the findings
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# We apply the function to all the examples in our train and test datasets
dataset['train'] = dataset['train'].map(preprocess_function, batched=True)
dataset['test']  = dataset['test'].map(preprocess_function, batched=True)

# Remove the original columns
dataset['train'] = dataset['train'].remove_columns(["input"])
dataset['test']  = dataset['test'].remove_columns(["input"])

# Write out the arguments
MODEL_OUT_NAME = f"{MODEL_NAME}_{DATASET_NAME}"

train_args = Seq2SeqTrainingArguments(
    f"models/{MODEL_OUT_NAME}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    include_inputs_for_metrics=True,
    report_to="wandb"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

# Create the Trainer and train
trainer = Seq2SeqTrainer(
    model = None, # model
    args=train_args,
    model_init=model_init,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="wandb",
    hp_space=wandb_hp_space,
    n_trials=10,
)
print('########### BEST TRIAL ###########')
print(best_trial)
print('##################################')

# trainer.train()

# Run evaluate to obtain the model's performance on the test dataset 
# trainer.evaluate()