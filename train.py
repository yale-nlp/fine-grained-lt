import argparse
import math
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
    def clean(self, s):
        """Replace all special tokens with the empty string

        Args:
            s (str): Original string

        Returns:
            str: Cleaned string
        """
        for t in self.tokenizer.all_special_tokens_extended:
            s = s.replace(str(t), "")
        return s

    def get_entities(self, input, ner_model_lst, linker_lst=None):

        SEMTYPES = ["T023","T028","T046","T047","T048",
                    "T059","T060","T061","T074","T109",
                    "T116","T121","T122","T123","T125",
                    "T129","T184","T191","T195"]

        output_entities = set()

        if type(ner_model_lst) is not list:
            ner_model_lst = [ner_model_lst]
            linker_lst    = [linker_lst]

        for (ner_model, linker) in zip(ner_model_lst, linker_lst):
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
                output_entities.update(set(filtered_entities))
            else:
                output_entities.update(set([str(e) for e in entity_lst]))

        return output_entities
        
    def shift_tokens_right(self, input_ids, pad_token_id):
        """Shift input ids one token to the right, and wrap the last
        non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def compute_entailment_mask(self, inputs, targets):
        # Decode and clean the inputs
        decoded_inp = self.tokenizer.batch_decode(inputs)
        decoded_inp = [self.clean(s) for s in decoded_inp] 

        # Decode and clean the targets
        decoded_tgt = self.tokenizer.batch_decode(targets) 
        decoded_tgt = [self.clean(s) for s in decoded_tgt] 

        # Use NER model to identify entities
        entities = [self.ner_model(s).ents for s in decoded_tgt] 

        # Distinguish the entailed vs non-entailed entities
        entity_strs = [[str(s) for s in lst] for lst in entities] 
        entailed_strs = [list(filter(lambda x: x.lower() in i.lower(), lst)) for (i, lst) in zip(decoded_inp, entity_strs)]
        non_entd_strs = [list(filter(lambda x: x.lower() not in i.lower(), lst)) for (i, lst) in zip(decoded_inp, entity_strs)]

        # Create a mask for the entailed entities
        entailed_strs = [" ".join([str(s) for s in lst]) for lst in entailed_strs]
        entailed_toks = self.tokenizer(entailed_strs, return_tensors="pt", padding=True)
        entailed_mask = torch.nn.functional.one_hot(
            entailed_toks["input_ids"], 
            num_classes = self.tokenizer.vocab_size
        )
        entailed_mask = entailed_mask.amax(axis=1)

        # Create a mask for the non-entailed entities
        non_entd_strs = [" ".join([str(s) for s in lst]) for lst in non_entd_strs] 
        non_entd_toks = self.tokenizer(non_entd_strs, return_tensors="pt", padding=True)
        non_entd_mask = torch.nn.functional.one_hot(
            non_entd_toks["input_ids"], 
            num_classes = self.tokenizer.vocab_size
        )
        non_entd_mask = -1*non_entd_mask.amax(axis=1)

        # entity_strs = [" ".join([str(s) for s in lst]) for lst in entities] # Concatenate those entities for tokenization
        # entity_toks = self.tokenizer(entity_strs, return_tensors="pt", padding=True) # Tokenize
        # entity_mask = torch.nn.functional.one_hot(
        #     entity_toks["input_ids"], 
        #     num_classes = self.tokenizer.vocab_size
        # )
        # entity_mask = entity_mask.amax(axis=1)

        # Remove any special IDs from the mask
        entity_mask = entailed_mask + non_entd_mask
        entity_mask[:, self.tokenizer.all_special_ids] = 0

        return entity_mask

    def compute_entity_mask(self, targets):

        # Decode and clean the targets
        decoded_tgt = self.tokenizer.batch_decode(targets) 
        decoded_tgt = [self.clean(s) for s in decoded_tgt] 

        # Use NER model to identify entities
        entities = [self.get_entities(s, self.ner_model, self.linker) for s in decoded_tgt] 
        entity_strs = [[str(s) for s in lst] for lst in entities] 

        # Create a mask for the entities
        entity_mask_lst = []
        for lst in entity_strs:
            if len(lst)>0:
                ent_toks = tokenizer(lst, return_tensors="pt", padding=True)
                sent_ent_mask = torch.nn.functional.one_hot(
                    ent_toks["input_ids"], 
                    num_classes = self.tokenizer.vocab_size
                )
                sent_ent_mask = sent_ent_mask.sum(axis=0).sum(axis=0)
                sent_ent_mask = 1*(sent_ent_mask>0)
                entity_mask_lst.append(sent_ent_mask)
            else:
                entity_mask_lst.append(torch.zeros(self.tokenizer.vocab_size))
        entity_mask = torch.vstack(entity_mask_lst)

        # Remove any special IDs from the mask
        entity_mask[:, self.tokenizer.all_special_ids] = 0

        return entity_mask
    
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

        return -1.0 * weighted_probs.sum(axis=-1).sum(axis=-1)

    def rejection_loss(
        self,
        logits,
        target,
        epsilon,
        unk_idx,
        ignore_index=None,
        reduce=True,
        mask=None,
        alpha=1.0
    ):
        batch_size, seq_len, vocab_size = logits.shape
        lprobs = torch.log(torch.softmax(logits, dim=-1))

        # if target.dim() == lprobs.dim() - 1:
        #     target = target.unsqueeze(-1)
        # nll_loss = -logits.gather(dim=-1, index=target)
        nll_loss = F.cross_entropy(
            logits.view(-1, vocab_size), 
            target.view(-1), 
            reduction='none'
            )
        nll_loss = nll_loss.view(batch_size, seq_len, 1)
        
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        # ================== calculate rejection loss ==================
        # rej_prob = torch.exp(lprobs[:, unk_idx]).unsqueeze(-1) 

        # Modified here, get logit on the unk_token (3rd dim)
        rej_prob = torch.exp(lprobs[:, :, unk_idx]).unsqueeze(-1) 

        if mask is not None:
            # Take entity mask; unsqueeze from (batch_size x vocab_size)
            # to (batch_size x seq_len x vocab_size)
            mask = mask[:, :vocab_size].unsqueeze(1).expand(batch_size, seq_len, vocab_size).eq(0)
            # Mask out only in the targets 
            mask = mask * torch.nn.functional.one_hot(target, num_classes=vocab_size)
            # Squeeze into the seq_len dimension
            mask = mask.max(axis=-1, keepdim=True).values 
            # Turn mask into bool
            mask = mask == True
            # Mask the rej_probs
            keep_prob = (1. - rej_prob).masked_fill(mask, 1.0)  # 0: non-entity
        else:
            keep_prob = 1. - rej_prob
        assert keep_prob.shape == nll_loss.shape, \
            "nll_loss: {}; keep_prob: {}".format(nll_loss.shape, keep_prob.shape)    

        # Essentially masks out the tokens which we don't wanna keep
        # Done using the keep_prob logic, but instead of just 1 and 0,
        # keep_prob is continuous – hence it's a "soft" masking
        rej_loss = keep_prob * (nll_loss + torch.log(keep_prob))
        rej_regularizer = -alpha * torch.log(keep_prob)
        nll_loss = rej_loss + rej_regularizer

        rej_smooth_loss = keep_prob * (smooth_loss + torch.log(keep_prob))
        smooth_loss = rej_smooth_loss + rej_regularizer
        # ===============================================================

        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.mean(axis=-1) # sum()
            smooth_loss = smooth_loss.mean(axis=-1) # sum()
        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models
        return the loss in the first element.
        Subclass and override for custom behavior.
        """
        self.iteration += 1
        if "meets_cutoff" in inputs:
            meets_cutoff = inputs.pop("meets_cutoff")
        else:
            meets_cutoff = None

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if args.loss_type != "standard" and self.is_in_train and (self.iteration > self.loss_cutoff):
            
            # Retrieve logits and labels
            logits = outputs["logits"]
            labels = inputs["labels"].to(logits.device)
            batch_size, seq_len, vocab_size = logits.size()

            # Compute cross entropy loss
            labels = labels.to(logits.device)
            nll_loss = F.cross_entropy(
                logits.view(-1, vocab_size), 
                labels.view(-1), 
                reduction='none')
            
            # Reshape NLL loss to bs x seq_len
            if (nll_loss.dim() == 1) and (batch_size > 1):
                # print(f"Batch size > 1, reshaping from {nll_loss.shape}")
                nll_loss = nll_loss.view(batch_size, seq_len)
                # print(f"To: {nll_loss.shape}")
            if (nll_loss.dim() == 1) and (batch_size == 1):
                # print(f"Batch size = 1, reshaping from {nll_loss.shape}")
                nll_loss = nll_loss.unsqueeze(0)
                # print(f"To: {nll_loss.shape}")

            # Rejection Loss
            if args.loss_type[:3] == "rej":
                # Compute entity_mask
                entity_mask = self.compute_entity_mask(labels)
                entity_mask = entity_mask.to(logits.device)

                loss, _ = self.rejection_loss(
                    logits = logits,
                    target = labels,
                    epsilon = 0.1,
                    unk_idx = self.tokenizer.unk_token_id,
                    ignore_index = None, # self.tokenizer.pad_token_id,
                    reduce = True,
                    mask = entity_mask,
                    alpha = 1.0
                )                

            if args.loss_type[:4] == "copy":
                # One hot encode the inputs (bs x seq_len_input x vocab) and labels (bs x seq_len_label x vocab)
                inputs_mask = torch.nn.functional.one_hot(inputs['input_ids'], num_classes=vocab_size)
                labels_mask = torch.nn.functional.one_hot(inputs['labels'], num_classes=vocab_size)
                # Reshape the inputs mask to match the labels sequence length
                inputs_mask = inputs_mask.max(axis=1, keepdim=True).values.expand(batch_size, labels_mask.shape[1], vocab_size)
                # Multiply input and label mask, then sum to get (bs x seq_len_labels), with 1 if it's copied, 0 otherwise
                labels_mask = labels_mask * inputs_mask
                
                # Only keep the 1's if the tokens correspond to entities
                is_ent_mask = self.compute_entity_mask(inputs["input_ids"])[:, :vocab_size].to(labels_mask.device)
                is_ent_mask = is_ent_mask.unsqueeze(1).expand(batch_size, labels_mask.shape[1], vocab_size)
                labels_mask = labels_mask * is_ent_mask

                # Sum across the vocab dimension, to get a (bs x seq_len target), then one hot encode to (bs x seq_len x 2)
                labels_mask = labels_mask.max(axis=-1).values
                labels_mask = torch.nn.functional.one_hot(labels_mask, num_classes=2).float()

                # Compute BCE Loss between predicted copy probability and copy label
                copy_pred = torch.matmul(logits, self.copy_linear_layer)
                copy_loss = torch.nn.functional.binary_cross_entropy_with_logits(copy_pred, labels_mask.float())

                wandb.log({"copy_loss": float(copy_loss)})
                
                loss = nll_loss.mean() + self.copy_weight*copy_loss

            if args.loss_type[:3] == "tok":
                
                source = inputs["input_ids"].to(logits.device)

                # Compute entailment mask: bs x vocab_size
                entailment_mask = self.compute_entailment_mask(source, labels)
                # To device
                entailment_mask = entailment_mask.to(logits.device)
                # Get the entailment signs for the tokens on the labels
                # i.e. for each token in the label, get its entailment sign
                # shape of the mask is from bs x vocab_size -> bs x seq_len
                entailment_mask = entailment_mask[:, labels].squeeze(1)
                # Mask out entities which are NOT entailed
                entailment_mask = 1*entailment_mask.ne(-1)

                loss = torch.mean(nll_loss * entailment_mask, axis=-1)
            
            if args.loss_type[:2] == "mi" and args.loss_type != "mi_lt":
                
                source = inputs["input_ids"]

                # Compute MI
                baseline_outputs = model(**{"input_ids": inputs["labels"]})
                baseline_logits  = baseline_outputs["logits"]
                mi_tensor = logits - baseline_logits

                # If use_generated_not_labels, we apply MI penalty to the entities in the output tokens
                # Otherwise, we apply MI penalty to the labels
                label_type = "labels"
                if label_type == "generated": 
                    # Generate logits indices mask to determine generated words
                    logits_indices = torch.argmax(logits, dim=-1)
                    target_mask = torch.nn.functional.one_hot(
                        logits_indices, num_classes=vocab_size
                    ) 
                    entailment_mask = self.compute_entailment_mask(source, logits_indices)
                elif label_type == "labels":
                    # Apply the MI loss only to the targets
                    target_mask = torch.nn.functional.one_hot(
                        labels, 
                        num_classes = vocab_size
                    )
                    # Also, the MI loss is positive if the target is not entailed
                    # and negative if the target is entailed
                    entailment_mask = self.compute_entailment_mask(source, labels)
                else:
                    # Compute target mask from generated tokens
                    logits_indices = torch.argmax(logits, dim=-1)
                    target_mask_gen = torch.nn.functional.one_hot(
                        logits_indices, num_classes=vocab_size
                    ) 
                    # Compute target mask from labels
                    target_mask_lab = torch.nn.functional.one_hot(
                        labels, 
                        num_classes = vocab_size
                    )
                    # Combine to get final target mask
                    target_mask = 1*((target_mask_gen + target_mask_lab) > 0)

                    # Compute entailment mask from both the generated tokens and labels
                    concat_indices = torch.cat([logits_indices, labels], axis=1)
                    entailment_mask = self.compute_entailment_mask(source, concat_indices)

                # Post-process the entailment mask:
                # Cut to vocab size
                entailment_mask = entailment_mask[:, :vocab_size] 
                # Expand across sequence length dimension
                entailment_mask = entailment_mask.unsqueeze(1)\
                                        .expand(batch_size, seq_len, vocab_size) 
                
                # Determine entailed vs non-entailed entities
                entailed_mask = 1.0*(entailment_mask.clone().eq(1))
                nonentld_mask = 1.0*(entailment_mask.clone().eq(-1))

                entailed_mask = entailed_mask.to(mi_tensor.device)
                nonentld_mask = nonentld_mask.to(mi_tensor.device)

                # So, the MI tensor is the MI value x target_mask x entailment_mask
                # Sum across the sequence length and vocab dimensions
                # We end up with an MI tensor of length batch_size
                mi_entailed = (mi_tensor * target_mask * entailed_mask).sum(axis=-1).mean(axis=-1)
                mi_nonentld = (mi_tensor * target_mask * nonentld_mask).sum(axis=-1).mean(axis=-1)

                # We want to MAXIMIZE the entailed MI
                # We can do this by setting its loss to exp(-x)
                mi_entailed_loss = torch.exp(-1.0*mi_entailed)

                # We want to MINIMIZE the non-entailed MI
                # We can do this by setting its loss to exp(x)
                mi_nonentld_loss = torch.exp(mi_nonentld)

                wandb.log({"mi_entailed":    float(mi_entailed_loss.mean()),
                           "mi_nonentailed": float(mi_nonentld_loss.mean())})
                loss = nll_loss.mean(axis=-1) + (self.mi_weight * (mi_entailed_loss + mi_nonentld_loss))

            if (args.loss_type[:2] == "ul") or (args.loss_type == "mi_ul"):
                
                # Generate UL weights
                ul_weight = torch.zeros((batch_size, 
                                        seq_len, 
                                        vocab_size)).float().cuda()

                selective_penalty   = True
                readability_penalty = True
                exclude_entities    = "_ent" in args.loss_type

                # Create entity mask if we are excluding entities
                if exclude_entities:
                    # Generate entity mask (bs x seq_len x vocab_size), which is 1 if
                    # the token is an entity present in the INPUTS, 0 otherwise

                    # Identify which tokens are entities in both labels and generated
                    entity_mask = self.compute_entity_mask(inputs["input_ids"]).to(logits.device)

                    # Cut to vocab size
                    entity_mask = entity_mask[:, :vocab_size] 

                    # Expand across sequence length dimension
                    entity_mask = entity_mask.unsqueeze(1)\
                                    .expand(batch_size, seq_len, vocab_size) 
                    
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
                # Applying this penalizes ONLY the generated words by their complexity
                if readability_penalty:
                    read_mask = self.ul_weights
                    read_mask = (
                        read_mask.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(batch_size, seq_len, vocab_size)
                        .clone()
                    )

                    if exclude_entities:
                        # Find tokens which are NOT entities and keep the UL penalty on those
                        # Zero out the penalty on entity words
                        ul_weight += self.lambda_read * read_mask * logits_indices_mask * 1*(entity_mask.eq(0))
                    else:
                        ul_weight += self.lambda_read * read_mask * logits_indices_mask

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
                    if exclude_entities:
                        # Find tokens which ARE entities and keep the hallucination penalty on those
                        # Zero out non-entity words
                        ul_weight += self.lambda_const * neg_indices_mask * logits_indices_mask * 1*(entity_mask.eq(1))
                    else:
                        # Penalize these non-appearing tokens with a fixed weight
                        ul_weight += self.lambda_const * neg_indices_mask * logits_indices_mask

                ul_loss = self.unlikelihood_loss(
                    decoder_input_ids=labels, logits=logits, weight_mask=ul_weight
                )

                # Add in the UL loss to the orig loss
                if args.loss_type == "mi_ul":
                    # add in the UL loss to the existing MI loss
                    # recall: mi_ul would have computed loss under "mi"
                    loss += ul_loss 
                else:
                    loss = nll_loss.mean(axis=-1) + ul_loss

            # if args.loss_type == "mi_lt":
            #     baseline_logits_clone = self.baseline_logits.clone()[:seq_len, :]
            #     baseline_logits_clone = baseline_logits_clone.unsqueeze(0).expand(
            #         batch_size, seq_len, vocab_size
            #         )
            #     mi_tensor = logits - baseline_logits_clone
            #     mi_tensor = torch.nn.functional.one_hot(
            #         logits.argmax(axis=-1), 
            #         num_classes=vocab_size
            #         ) * mi_tensor
            #     mi = mi_tensor.sum(axis=-1).mean(axis=-1)
            #     loss = nll_loss.mean(axis=-1)

            if args.loss_type[:6] == "cutoff":
                assert meets_cutoff is not None
                loss = nll_loss.mean(axis=-1)
                item_weights = torch.ones(meets_cutoff.shape[0]).to(self.model.device)
                item_weights[meets_cutoff == 1] = self.meets_cutoff_weight
                item_weights[meets_cutoff == 0] = (self.not_meets_cutoff_weight*math.exp(-0.1*self.step_counter))
                loss *= item_weights

            if args.loss_type == "mi_lt":
                # Compute the MI to be used for the cutoff
                baseline_logits_clone = self.baseline_logits.clone()[:seq_len, :]
                baseline_logits_clone = baseline_logits_clone.unsqueeze(0).expand(
                    batch_size, seq_len, vocab_size
                    )
                
                mi_tensor = logits - baseline_logits_clone

                entity_mask = self.compute_entity_mask(labels).to(logits.device)
                entity_mask = entity_mask[:, :vocab_size] 
                entity_mask = entity_mask.unsqueeze(1)\
                                .expand(batch_size, seq_len, vocab_size) 
                
                label_mask = torch.nn.functional.one_hot(
                    labels, 
                    num_classes=vocab_size
                    )
                
                mi = mi_tensor * entity_mask * label_mask
                mi = mi.sum(axis=-1)
                mi = mi.sum(axis=-1) / (1*(mi > 0)).sum(axis=-1)

            if args.loss_type == "max_lt":
                # Compute the max LT to be used for the cutoff

                # Identify which tokens are entities in labels
                # labels_copy = labels.clone().detach()
                # labels_copy[labels_copy == -100] = self.tokenizer.pad_token_id
                entity_mask = self.compute_entity_mask(labels).to(logits.device)
                entity_mask = entity_mask[:, :vocab_size] 
                entity_mask = entity_mask.unsqueeze(1)\
                                .expand(batch_size, seq_len, vocab_size) 

                label_mask = torch.nn.functional.one_hot(
                    labels, 
                    num_classes=vocab_size
                    )
                # Create the mask which is 1 if a token is an entity
                # and 0 otherwise
                label_entity_mask = entity_mask * label_mask
                label_entity_mask = label_entity_mask.amax(axis=-1)

                # Change: Sum the loss across all the entities
                # max_nll = (nll_loss * label_entity_mask).amax(-1)
                max_nll = nll_loss * label_entity_mask
                max_nll = max_nll.sum(axis=-1) # / (1*(max_nll > 0)).sum(axis=-1)
            
            if (args.loss_type[:2] == "lt") or ("_lt" in args.loss_type):

                # Use NLL as the loss
                loss = nll_loss.mean(axis=-1)
                loss = loss.view(-1, batch_size)

                # The dropper returns a mask of 0s where data should be dropped.
                if args.loss_type == "mi_lt":
                    mask = self.loss_dropper(mi)
                elif args.loss_type == "max_lt":
                    mask = self.loss_dropper(max_nll)
                else:
                    mask = self.loss_dropper(loss)  
                loss = loss * mask  # Mask out the high losses
            
            # Regardless of loss type, take the mean across the samples in the batch
            loss = loss.mean()  # Aggregate

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

        # Specify 
        trainer.iteration = 0
        trainer.loss_cutoff = args.loss_cutoff
        
        # Add loss specific arguments to the trainer
        if (args.loss_type[:2] == "ul") or ("_ul" in args.loss_type):
            with open(f"{ROOT_PATH}/fk_weights.pkl", "rb") as f:
                ul_weights = pickle.load(f)
            ul_weights = list(map(lambda x: max(x, 0.0), ul_weights))
            trainer.ul_weights = torch.tensor(ul_weights).float().cuda()
            trainer.lambda_read  = 5e-4 
            trainer.lambda_const = 1.5e-4
            if "_ent" in args.loss_type:
                ner_model_web = spacy.load("en_core_web_lg")
                ner_model_sci = spacy.load("en_core_sci_lg")
                ner_model_sci.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": "umls"},
                )
                linker_sci = ner_model_sci.get_pipe("scispacy_linker")
                trainer.ner_model = [ner_model_sci, ner_model_web]
                trainer.linker    = [linker_sci, None]

        if args.loss_type[:6] == "cutoff":
            trainer.meets_cutoff_weight = 1.0
            trainer.not_meets_cutoff_weight = 1.0
            trainer.step_counter = 0
            trainer._signature_columns = ["input_ids","attention_mask","labels","meets_cutoff"]

        if args.loss_type[:3] == "tok":
            trainer.ner_model = spacy.load("en_core_web_lg")

        if args.loss_type[:3] == "rej":
            trainer.ner_model = [spacy.load("en_core_web_lg")]
            trainer.linker    = [None]

        if args.loss_type[:4] == "copy":
            ner_model_web = spacy.load("en_core_web_lg")
            ner_model_sci = spacy.load("en_core_sci_lg")
            ner_model_sci.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
            linker_sci = ner_model_sci.get_pipe("scispacy_linker")
            trainer.ner_model = [ner_model_sci, ner_model_web]
            trainer.linker    = [linker_sci, None]

            linear_size = trainer.tokenizer.vocab_size - 1 if args.model == "bart_xsum" else trainer.tokenizer.vocab_size
            trainer.copy_linear_layer = torch.rand((linear_size, 2), 
                                        requires_grad=True).to(trainer.model.device)
            trainer.copy_weight = 1e-2

        if args.loss_type == "max_lt":
            ner_model_web = spacy.load("en_core_web_lg")
            ner_model_sci = spacy.load("en_core_sci_lg")
            ner_model_sci.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
            linker_sci = ner_model_sci.get_pipe("scispacy_linker")
            trainer.ner_model = [ner_model_sci, ner_model_web]
            trainer.linker    = [linker_sci, None]

        if args.loss_type[:2] == "mi":
            ner_model_web = spacy.load("en_core_web_lg")
            ner_model_sci = spacy.load("en_core_sci_lg")
            ner_model_sci.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
            linker_sci = ner_model_sci.get_pipe("scispacy_linker")
            trainer.ner_model = [ner_model_sci, ner_model_web]
            trainer.linker    = [linker_sci, None]
            trainer.mi_weight = 1.0
        
        if args.loss_type == "mi_lt":
            trainer.baseline_logits = torch.load(
                f"/home/lyf6/simplification-project/nli/logits/{args.dataset}_baseline_logits.pt"
                ).to(trainer.model.device)

        # If loss truncation is applied, add a loss_dropper module
        if (args.loss_type[:2] == "lt") or ("_lt" in args.loss_type):
            trainer.loss_dropper = LossDropper(
                dropc=0.4,
                min_count=500,
                recompute=500,
                verbose=True
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
    elif "_tok" in args.checkpoint:
        LOSS_TYPE_NAME = "_tok"
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
