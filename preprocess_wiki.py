import argparse
import json
import os
import pickle
import spacy
import shutil

from utils import add_context


def write_json(output_json, path):
    json_object = json.dumps(output_json, indent=4)
    with open(path, "w") as outfile:
        outfile.write(json_object)


# Get dataset from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
args = parser.parse_args()
print(f"Using dataset: {args.dataset}")

if args.dataset in ["asset", "turkcorpus"]:
    ner_model = spacy.load("en_core_web_lg")
    linker = None
    kb = "wordnet_wikipedia"
elif args.dataset in ["radiology_indiv", "radiology_full", "radiology", "cochrane"]:
    import scispacy
    from scispacy.linking import EntityLinker

    ner_model = spacy.load("en_core_sci_lg")
    ner_model.add_pipe(
        "scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}
    )
    linker = ner_model.get_pipe("scispacy_linker")
    kb = "umls"
else:
    assert False

# Read files
input_dict = {"name": args.dataset, "input": f"data/{args.dataset}_multiple.json"}
data = json.load(open(input_dict["input"]))

train_input_lst = list(map(lambda d: d["input"], data["train"]))
train_label_lst = list(map(lambda d: d["labels"], data["train"]))

test_input_lst = list(map(lambda d: d["input"], data["test"]))
test_label_lst = list(map(lambda d: d["labels"], data["test"]))

# Add context
if os.path.exists(f"misc/train_{input_dict['name']}_{kb}.pkl") and os.path.exists(
    f"misc/test_{input_dict['name']}_{kb}.pkl"
):
    # If they exist, read context files from past run
    with open(f"misc/train_{input_dict['name']}_{kb}.pkl", "rb") as input_file:
        train_wiki_input_lst = pickle.load(input_file)
    with open(f"misc/test_{input_dict['name']}_{kb}.pkl", "rb") as input_file:
        test_wiki_input_lst = pickle.load(input_file)
else:
    # Create a directory to save temporary files
    os.makedirs(f"data/temp/{input_dict['name']}", exist_ok=True)

    # For each item, add context and temporarily save in a pickle file
    for idx, item in enumerate(train_input_lst):
        item_replaced = add_context(item, ner_model, kb, linker)
        with open(f"data/temp/{input_dict['name']}/train_{idx}.pkl", "wb") as handle:
            pickle.dump(item_replaced, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for idx, item in enumerate(test_input_lst):
        item_replaced = add_context(item, ner_model, kb, linker)
        with open(f"data/temp/{input_dict['name']}/test_{idx}.pkl", "wb") as handle:
            pickle.dump(item_replaced, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_wiki_input_lst = [
        pickle.load(open(f"data/temp/{input_dict['name']}/train_{idx}.pkl", "rb"))
        for idx in range(len(train_input_lst))
    ]
    test_wiki_input_lst = [
        pickle.load(open(f"data/temp/{input_dict['name']}/test_{idx}.pkl", "rb"))
        for idx in range(len(test_input_lst))
    ]

    with open(f"misc/train_{input_dict['name']}_{kb}.pkl", "wb") as f:
        pickle.dump(train_wiki_input_lst, f)
    with open(f"misc/test_{input_dict['name']}_{kb}.pkl", "wb") as f:
        pickle.dump(test_wiki_input_lst, f)
    shutil.rmtree(f"data/temp/{input_dict['name']}", ignore_errors=True)

# Filter datasets by number of entities found
train_all_filter = [
    i for i in range(len(train_wiki_input_lst)) if train_wiki_input_lst[i][1] == "all"
]
test_all_filter = list(range(len(test_wiki_input_lst)))

train_some_filter = [
    i for i in range(len(train_wiki_input_lst)) if train_wiki_input_lst[i][1] != "none"
]
test_some_filter = list(range(len(test_wiki_input_lst)))

input_lst_all = (train_all_filter, test_all_filter, "all")
input_lst_some = (train_some_filter, test_some_filter, "some")

# Assemble output JSON
for train_idx, test_idx, filter_type in [input_lst_all, input_lst_some]:
    train_input_lst_ = [train_wiki_input_lst[i][0] for i in train_idx]
    train_label_lst_ = [train_label_lst[i] for i in train_idx]
    test_input_lst_ = [test_wiki_input_lst[i][0] for i in test_idx]
    test_label_lst_ = [test_label_lst[i] for i in test_idx]

    # Write individual json for each row, first using one reference per input (for training)
    train_json, test_json = [], []
    for (input, labels) in zip(train_input_lst_, train_label_lst_):
        train_json.extend([{"input": input, "labels": [l]} for l in labels])
    for (input, labels) in zip(test_input_lst_, test_label_lst_):
        test_json.extend([{"input": input, "labels": [l]} for l in labels])
    output_json_indiv = {"train": train_json, "test": test_json}

    # Output the file
    write_json(
        output_json_indiv, f"data/{input_dict['name']}_context_{filter_type}.json"
    )

    # Then multiple references per input (for evaluation)
    train_json = [
        {"input": input, "labels": labels}
        for (input, labels) in zip(train_input_lst_, train_label_lst_)
    ]
    test_json = [
        {"input": input, "labels": labels}
        for (input, labels) in zip(test_input_lst_, test_label_lst_)
    ]
    output_json_mult = {"train": train_json, "test": test_json}

    # Output the file
    write_json(
        output_json_mult,
        f"data/{input_dict['name']}_context_{filter_type}_multiple.json",
    )
