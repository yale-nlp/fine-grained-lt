import time
import openai
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import torch

SEMTYPES = [
    "T023",
    "T028",
    "T046",
    "T047",
    "T048",
    "T059",
    "T060",
    "T061",
    "T074",
    "T109",
    "T116",
    "T121",
    "T122",
    "T123",
    "T125",
    "T129",
    "T184",
    "T191",
    "T195",
]

def get_entities(input, ner_model, linker=None):
    entity_lst = ner_model(input).ents

    if "scispacy_linker" in ner_model.pipe_names:
        filtered_entities = []
        for e in set(entity_lst):
            if len(e._.kb_ents) > 0:
                umls_ent_id, _ = e._.kb_ents[0]  # Get top hit from UMLS
                umls_ent  = linker.kb.cui_to_entity[umls_ent_id]  # Get UMLS entity
                umls_semt = umls_ent[3]
                if any([t in SEMTYPES for t in umls_semt]):
                    e = str(e)
                    if e not in filtered_entities:
                        filtered_entities.append(e)
        return filtered_entities
    else:
        return list(set([str(e) for e in entity_lst]))
    
def create_ent_entailment_mask(reference, entities, tokenizer):
        
    # Find the entailed and non-entailed entities
    non_entd_strs = [s for s in entities if s.lower() not in reference.lower()]
    entd_strs = list(set(entities).difference(set(non_entd_strs)))

    # Create a mask for the entailed entities
    if len(entd_strs)>0:
        entd_toks = tokenizer(
            entd_strs, 
            return_tensors="pt", 
            padding=True
            )
        entd_mask = torch.nn.functional.one_hot(
            entd_toks["input_ids"], 
            num_classes = tokenizer.vocab_size
        )
        # Put a 1 in the token if it is present in the label but not entailed
        entd_mask = entd_mask.sum(axis=0).sum(axis=0)
        entd_mask = 1*(entd_mask>0)
    else:
        entd_mask = torch.zeros(tokenizer.vocab_size)

    # Create a mask for the non-entailed entities
    if len(non_entd_strs)>0:
        non_entd_toks = tokenizer(non_entd_strs, 
                                return_tensors="pt", 
                                padding=True)
        non_entd_mask = torch.nn.functional.one_hot(
            non_entd_toks["input_ids"], 
            num_classes = tokenizer.vocab_size
        )
        # Put a 1 in the token if it is present in the label but not entailed
        non_entd_mask = non_entd_mask.sum(axis=0).sum(axis=0)
        non_entd_mask = -1*(non_entd_mask>0)
    else:
        non_entd_mask = torch.zeros(tokenizer.vocab_size)

    # Remove any special IDs from the mask
    mask = entd_mask + non_entd_mask
    mask[tokenizer.all_special_ids] = 0

    return mask

def create_sent_entailment_mask(sent_lst, nli_lst, tokenizer, pad_to=0):
    sent_lengths = tokenizer(
        sent_lst, 
        padding=True, 
        return_tensors="pt")["attention_mask"].sum(axis=1) - 2
    
    token_nli = [0]
    for i in range(len(sent_lengths)):
        token_nli.extend([nli_lst[i]] * int(sent_lengths[i]))
    token_nli.append(0)

    if pad_to > 0:
        token_nli.extend([0]*(pad_to - len(token_nli)))

    output = torch.Tensor(token_nli).type(torch.int64)
    return output
    
def reshape_vocab_mask_to_sequence_mask(mask, target):
    # If input is just (vocab_size), unsqueeze first dimension
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    # Take entity mask; unsqueeze from (batch_size x vocab_size)
    # to (batch_size x seq_len x vocab_size)
    batch_size, vocab_size = mask.shape
    seq_len = target.shape[1]

    mask = mask.unsqueeze(1).expand(batch_size, seq_len, vocab_size)

    # Mask out only in the targets 
    mask = mask * torch.nn.functional.one_hot(target, num_classes=vocab_size)

    # Squeeze into the seq_len dimension
    mask = mask.sum(axis=-1)
    mask[mask > 0] = 1
    mask[mask < 0] = -1
    
    return mask

def encode(example, tokenizer):
    """This function takes a batch of samples,
    and tokenizes them into IDs for the model."""
    # Tokenize the Findings (the input)
    input_str = example["input"]
    model_inputs = tokenizer(
        input_str, 
        max_length=768, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )

    # Tokenize the Impressions (the output)
    labels = tokenizer(
        example["labels"],
        max_length=768,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Set the label as the token ids (i.e. the vocab IDs) of the findings
    model_inputs["labels"] = labels["input_ids"]

    # Add anything else here and return it as an additional output
    # HERE!

    return model_inputs

def g_nli(sources: List[str], 
        predictions: List[str], 
        model: str, 
        **kwargs):

    openai.api_key_path = "openai_key"

    result = []
    for document, summary in zip(sources, predictions):
        time.sleep(0.3)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Your task is to rate the summary on one metric.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Human Evaluation of Text Summarization Systems: \n"
                            "Factual Consistency: Does the summary have untruthful or "
                            "misleading facts that are not supported by the source text? \n"
                            f"Source Text: {document} \n"
                            f"Summary: {summary} \n"
                            "Does the summary contain factual inconsistencies? \n"
                            "Answer: "
                            # "If yes, why: "
                        ),
                    },
                ],
                **kwargs,
            )
        except Exception as e:
            response = {}
            print(e)
        result.append(response)
    return result


def get_example_nli(example):
    # Unpack as source and predictions, just use one sentence
    input = [example['input']]
    label_sent_lst = [example['labels'][0]]

    # Deprecated
    # label_sent_lst = sent_tokenize(example['labels'][0])
    # input = input*len(label_sent_lst)

    # Use NLI to evaluate
    geval_dict = g_nli(
        input, 
        label_sent_lst, 
        model="gpt-4", 
        temperature=1
    )

    # Get text results
    nli_results = [result["choices"][0]["message"]["content"] \
                   for result in geval_dict]

    # Return predictions and NLI results
    return label_sent_lst, nli_results