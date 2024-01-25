from utils_nli import (
    get_example_nli, 
    encode, 
    create_sent_entailment_mask,
    create_ent_entailment_mask,
    reshape_vocab_mask_to_sequence_mask,
    get_entities
    )

import argparse
import os
import pandas as pd
import pickle
import spacy
import scispacy
import time
import torch
import torch.nn.functional as F

from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from scipy.stats import mannwhitneyu
from scispacy.linking import EntityLinker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict

device = torch.device("cuda")

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

model_checkpoint = args.checkpoint

# BART Base (don't use!)
# model_checkpoint = "models/BART_cochrane/7619j5uo/checkpoint-3568"
# model_checkpoint = "models/BART_medeasi/5kmaxv72/checkpoint-1397"
# model_checkpoint = "facebook/bart-large-xsum"

ROOT_PATH = "/home/lyf6/simplification-project"
DATASET_NAME = args.dataset
MODEL_NAME = "baseline" if model_checkpoint=="facebook/bart-large-xsum" else "trained"
LOGITS_PATH = f"{ROOT_PATH}/nli/logits/{DATASET_NAME}_{MODEL_NAME}_logits.pt"

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

# Load and preprocess the datasets
dataset = load_dataset(
    "json", 
    data_files=f"{ROOT_PATH}/data/{DATASET_NAME}.json",
    field="train")
dataset["test"] = load_dataset(
    "json", 
    data_files=f"{ROOT_PATH}/data/{DATASET_NAME}_multiple.json", 
    field="test"
)["train"]

# Load NER model
ner_model = spacy.load("en_core_sci_lg")
ner_model.add_pipe(
    "scispacy_linker",
    config={"resolve_abbreviations": True, "linker_name": "umls"},
)
linker = ner_model.get_pipe("scispacy_linker")

# Get baseline logits
if os.path.exists(LOGITS_PATH):
    baseline_logits = torch.load(LOGITS_PATH).to(device)
else:
    temp_logits_lst = []
    logits_lst = []

    for idx, example in enumerate(dataset["train"]):

        # Unpack batch
        batch = encode(example, tokenizer)
        batch = {k: batch[k].to(device) for k in batch}
        labels = batch["labels"]
    
        # Logits
        outputs = model(**batch)
        logits  = outputs["logits"].detach().cpu()
        temp_logits_lst.append(logits)

        if (idx % 1000 == 0) and (idx > 0):
            print(f"{idx}/{len(dataset['train'])}")
            logits_lst.append(torch.vstack(temp_logits_lst).mean(axis=0).squeeze(0))
            temp_logits_lst = []
    
        if (args.dataset == "cnn") and (idx >= 100000):
            break

    num_batches  = len(logits_lst)
    final_logits = sum(logits_lst)

    if len(temp_logits_lst) > 0:
        num_extra    = len(temp_logits_lst)
        temp_logits  = torch.vstack(temp_logits_lst).mean(axis=0).squeeze(0)
        final_logits = final_logits + ((num_extra/1000)*temp_logits)
        final_logits = final_logits / (num_batches + (num_extra/1000))
    else:
        final_logits = final_logits / num_batches

    baseline_logits = final_logits.to(device)
    torch.save(final_logits, LOGITS_PATH)

# Collect results
df = pd.DataFrame(columns=[
    'nll_mean', 
    'nli_flag_gpt_label', 
    'nli_flag_ent_label', 
    'nll_ent_0', 
    'nll_ent_1', 
    'nll_ent_-1', 
    'mi_mean', 
    'nli_flag_ent_output', 
    'mi_ent_0', 
    'mi_ent_1', 
    'mi_ent_-1'
    ])

for idx, example in enumerate(dataset["train"]):
    if idx%100==0:
        print(idx)
    
    if (args.dataset == "cnn") and (idx >= 100000):
        break
    
    # Unpack batch
    batch = encode(example, tokenizer)
    batch = {k: batch[k].to(device) for k in batch}
    labels = batch["labels"]
   
    # Logits
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs["logits"]
    batch_size, seq_len, vocab_size = logits.shape

    # NLL Loss
    nll_loss = F.cross_entropy(
            logits.view(-1, vocab_size), 
            labels.view(-1), 
            reduction='none'
            )
    nll_loss = nll_loss.view(seq_len).tolist()

    # NLL Mean Loss
    nll_mean = sum(nll_loss)/len(nll_loss)

    # Create entity-level NLI mask for label 
    # (1 if entailed, -1 if non-entailed, 0 if not entity)
    label_entities = get_entities(
        input=example["labels"][0], 
        ner_model=ner_model, 
        linker=linker
        )
    nli_ent_mask_label = create_ent_entailment_mask(
        reference=example["input"], 
        entities=label_entities, 
        tokenizer=tokenizer
        )
    nli_ent_mask_label = reshape_vocab_mask_to_sequence_mask(
        mask=nli_ent_mask_label,
        target=batch["labels"].cpu()
        )
    nli_ent_mask_label = nli_ent_mask_label.reshape(seq_len).tolist()

    # # Get sent-level NLI results
    # NLI_PATH = f"data/nli/cochrane/train/{idx}.pickle"
    # if os.path.exists(NLI_PATH):
    #     label_sents, nli_results = pickle.load(open(NLI_PATH, "rb"))
    # else:
    #     result = get_example_nli(example)
    #     label_sents, nli_results = result
    #     with open(NLI_PATH, 'wb') as handle:
    #         pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Create sent-level NLI Mask (1 if non-entailed, 0 if entailed)
    # label_sents = [" "+s if idx>0 else s for (idx, s) in enumerate(label_sents)]
    # nli_sent_lst = [1 if sent[:3]=="Yes" else 0 for sent in nli_results]
    # nli_sent_mask = create_sent_entailment_mask(
    #     sent_lst = label_sents, 
    #     nli_lst = nli_sent_lst, 
    #     tokenizer = tokenizer,
    #     pad_to = 768
    #     ).tolist()
    # nli_flag_gpt = 1 * any([i==1 for i in nli_sent_mask])
    
    # NLI Flag for labels (i.e. -1 if there's ANY non-entailment, 1 otherwise)
    # nli_flag_gpt_label = -1 if any([sent[:3]=="Yes" for sent in nli_results]) else 1
    nli_flag_ent_label = -1 if any([i==-1 for i in nli_ent_mask_label]) else 1

    # # MI
    # mi_tensor = logits - baseline_logits
    # mi_tensor = torch.nn.functional.one_hot(
    #     logits.argmax(axis=-1), # labels, # 
    #     num_classes=vocab_size
    #     ) * mi_tensor
    # mi_tensor = mi_tensor.sum(axis=-1)
    # mi_tensor = mi_tensor.reshape(seq_len).tolist()

    # # MI Mean
    # mi_mean = sum(mi_tensor)/len(mi_tensor)

    # Create entity-level NLI mask for output 
    # (1 if entailed, -1 if non-entailed, 0 if not entity)
    output_tokens = logits.argmax(axis=-1)
    output_entities = get_entities(
        input=tokenizer.batch_decode(output_tokens)[0], 
        ner_model=ner_model, 
        linker=linker
        )
    nli_ent_mask_output = create_ent_entailment_mask(
        reference=example["input"], 
        entities=output_entities, 
        tokenizer=tokenizer
        )
    nli_ent_mask_output = reshape_vocab_mask_to_sequence_mask(
        mask=nli_ent_mask_output,
        target=output_tokens.cpu()
        )
    nli_ent_mask_output = nli_ent_mask_output.reshape(seq_len).tolist()

    # NLI Flag for outputs (i.e. -1 if there's ANY non-entailment, 1 otherwise)
    nli_flag_ent_output = -1 if any([i==-1 for i in nli_ent_mask_output]) else 1

    # Collate results in dataframes
    temp_df = pd.DataFrame({"ent_mask": nli_ent_mask_label, "nll": nll_loss})
    temp_df_ent_nll  = temp_df.groupby("ent_mask").aggregate({"nll": "mean"}).reset_index()
    
    # temp_df = pd.DataFrame({"ent_mask": nli_ent_mask_output, "mi": mi_tensor})
    # temp_df_ent_mi   = temp_df.groupby("ent_mask").aggregate({"mi": "mean"}).reset_index()
    
    new_row = {
        "nll_mean":   nll_mean,
        # "nli_flag_gpt_label": nli_flag_gpt_label,
        "nli_flag_ent_label": nli_flag_ent_label,
        "nll_ent_0":  float(temp_df_ent_nll.loc[temp_df_ent_nll.ent_mask==0,    "nll"].iloc[0]) if len(temp_df_ent_nll.loc[temp_df_ent_nll.ent_mask==0])>0 else None,
        "nll_ent_1":  float(temp_df_ent_nll.loc[temp_df_ent_nll.ent_mask==1,    "nll"].iloc[0]) if len(temp_df_ent_nll.loc[temp_df_ent_nll.ent_mask==1])>0 else None,
        "nll_ent_-1": float(temp_df_ent_nll.loc[temp_df_ent_nll.ent_mask==-1,   "nll"].iloc[0]) if len(temp_df_ent_nll.loc[temp_df_ent_nll.ent_mask==-1])>0 else None,
        # "mi_mean":    mi_mean,
        "nli_flag_ent_output": nli_flag_ent_output,
        # "mi_ent_0":   float(temp_df_ent_mi.loc[temp_df_ent_mi.ent_mask==0,  "mi"].iloc[0]) if len(temp_df_ent_mi.loc[temp_df_ent_mi.ent_mask==0])>0 else None,
        # "mi_ent_1":   float(temp_df_ent_mi.loc[temp_df_ent_mi.ent_mask==1,  "mi"].iloc[0]) if len(temp_df_ent_mi.loc[temp_df_ent_mi.ent_mask==1])>0 else None,
        # "mi_ent_-1":  float(temp_df_ent_mi.loc[temp_df_ent_mi.ent_mask==-1, "mi"].iloc[0]) if len(temp_df_ent_mi.loc[temp_df_ent_mi.ent_mask==-1])>0 else None,
        }
    df.loc[len(df)] = new_row

df.to_csv(f"experiment_{DATASET_NAME}_{MODEL_NAME}_finetune.csv", index=False)