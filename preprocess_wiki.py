import json
from utils import replace_entities
import pickle
from utils import extract_entities
import argparse
import os


def read_json(path):
    with open(path, "r") as file:
        lines = file.readlines()
        lines = list(map(lambda x: x.strip("\n"), lines))
    return lines


def write_json(output_json, path):
    json_object = json.dumps(output_json, indent=4)
    with open(path, "w") as outfile:
        outfile.write(json_object)


ASSET_FILES = {
    "name": "asset",
    "train_input": "data/raw/asset/asset.valid.orig",
    "train_labels": [
        "data/raw/asset/asset.valid.simp.0",
        "data/raw/asset/asset.valid.simp.1",
        "data/raw/asset/asset.valid.simp.2",
        "data/raw/asset/asset.valid.simp.3",
        "data/raw/asset/asset.valid.simp.4",
        "data/raw/asset/asset.valid.simp.5",
        "data/raw/asset/asset.valid.simp.6",
        "data/raw/asset/asset.valid.simp.7",
        "data/raw/asset/asset.valid.simp.8",
        "data/raw/asset/asset.valid.simp.9",
    ],
    "test_input": "data/raw/asset/asset.test.orig",
    "test_labels": [
        "data/raw/asset/asset.test.simp.0",
        "data/raw/asset/asset.test.simp.1",
        "data/raw/asset/asset.test.simp.2",
        "data/raw/asset/asset.test.simp.3",
        "data/raw/asset/asset.test.simp.4",
        "data/raw/asset/asset.test.simp.5",
        "data/raw/asset/asset.test.simp.6",
        "data/raw/asset/asset.test.simp.7",
        "data/raw/asset/asset.test.simp.8",
        "data/raw/asset/asset.test.simp.9",
    ],
}
COCH_FILES = {
    "name": "cochrane",
    "train_input": "data/raw/cochrane/train.source",
    "train_labels": ["data/raw/cochrane/train.target"],
    "test_input": "data/raw/cochrane/test.source",
    "test_labels": ["data/raw/cochrane/test.target"],
}
RADR_FILES = {
    "name": "radiology",
    "train_input": "data/raw/radiology/chest.source",
    "train_labels": ["data/raw/radiology/chest.target"],
    "test_input": "data/raw/radiology/chest.source",
    "test_labels": ["data/raw/radiology/chest.target"],
}

TURK_FILES = {
    "name": "turkcorpus",
    "train_input": "data/raw/turkcorpus/tune.8turkers.tok.norm",
    "train_labels": [
        "data/raw/turkcorpus/tune.8turkers.tok.turk.0",
        "data/raw/turkcorpus/tune.8turkers.tok.turk.1",
        "data/raw/turkcorpus/tune.8turkers.tok.turk.2",
        "data/raw/turkcorpus/tune.8turkers.tok.turk.3",
        "data/raw/turkcorpus/tune.8turkers.tok.turk.4",
        "data/raw/turkcorpus/tune.8turkers.tok.turk.5",
        "data/raw/turkcorpus/tune.8turkers.tok.turk.6",
        "data/raw/turkcorpus/tune.8turkers.tok.turk.7",
    ],
    "test_input": "data/raw/turkcorpus/test.8turkers.tok.norm",
    "test_labels": [
        "data/raw/turkcorpus/test.8turkers.tok.turk.0",
        "data/raw/turkcorpus/test.8turkers.tok.turk.1",
        "data/raw/turkcorpus/test.8turkers.tok.turk.2",
        "data/raw/turkcorpus/test.8turkers.tok.turk.3",
        "data/raw/turkcorpus/test.8turkers.tok.turk.4",
        "data/raw/turkcorpus/test.8turkers.tok.turk.5",
        "data/raw/turkcorpus/test.8turkers.tok.turk.6",
        "data/raw/turkcorpus/test.8turkers.tok.turk.7",
    ],
}

# Get dataset from arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
args = parser.parse_args()
print(f"Using dataset: {args.dataset}")

if args.dataset == "asset":
    input_dict = ASSET_FILES
    bio_flag = False
elif args.dataset == "turkcorpus":
    input_dict = TURK_FILES
    bio_flag = False
elif args.dataset == "cochrane":
    input_dict = COCH_FILES
    bio_flag = True
elif args.dataset == "radiology":
    input_dict = RADR_FILES
    bio_flag = True
elif args.dataset in ["radiology_indiv", "radiology_full"]:
    input_dict = {"name": args.dataset, "input": f"data/{args.dataset}_multiple.json"}
    bio_flag = True
else:
    assert False

# Read files
if args.dataset in ["radiology_indiv", "radiology_full"]:
    data = json.load(open(input_dict["input"]))
    train_input_lst = list(map(lambda d: d["input"], data["train"]))
    train_vocab_lst = list(map(lambda d: d["vocab"], data["train"]))
    train_label_lst = [list(map(lambda d: d["labels"][0], data["train"]))]

    test_input_lst = list(map(lambda d: d["input"], data["test"]))
    test_vocab_lst = list(map(lambda d: d["vocab"], data["test"]))
    test_label_lst = [list(map(lambda d: d["labels"][0], data["test"]))]
else:
    train_input_lst = read_json(input_dict["train_input"])
    train_label_lst = [read_json(f) for f in input_dict["train_labels"]]
    test_input_lst = read_json(input_dict["test_input"])
    test_label_lst = [read_json(f) for f in input_dict["test_labels"]]

if os.path.exists(f"misc/test_{input_dict['name']}_lst.pkl") and os.path.exists(
    f"misc/test_{input_dict['name']}_lst.pkl"
):
    with open(f"misc/train_{input_dict['name']}_lst.pkl", "rb") as input_file:
        train_wiki_input_lst = pickle.load(input_file)
    with open(f"misc/test_{input_dict['name']}_lst.pkl", "rb") as input_file:
        test_wiki_input_lst = pickle.load(input_file)

else:
    if args.dataset in ["radiology_indiv", "radiology_full"]:
        # Add descriptions to entities from Wikipedia/Medline
        train_wiki_input_lst = list(
            map(
                lambda s, v: replace_entities(s, bio=bio_flag, entities=v),
                train_input_lst,
                train_vocab_lst,
            )
        )
        test_wiki_input_lst = list(
            map(
                lambda s, v: replace_entities(s, bio=bio_flag, entities=v),
                test_input_lst,
                test_vocab_lst,
            )
        )

    elif args.dataset == "cochrane":
        for idx, item in enumerate(train_input_lst):
            item_replaced = replace_entities(item, bio=bio_flag)
            with open(f"data/cochrane/train_{idx}.pkl", "wb") as handle:
                pickle.dump(item_replaced, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for idx, item in enumerate(test_input_lst):
            item_replaced = replace_entities(item, bio=bio_flag)
            with open(f"data/cochrane/test_{idx}.pkl", "wb") as handle:
                pickle.dump(item_replaced, handle, protocol=pickle.HIGHEST_PROTOCOL)

        train_wiki_input_lst = [
            pickle.load(open(f"data/cochrane/train_{idx}.pkl", "rb"))
            for idx in range(len(train_input_lst))
        ]
        test_wiki_input_lst = [
            pickle.load(open(f"data/cochrane/test_{idx}.pkl", "rb"))
            for idx in range(len(test_input_lst))
        ]

    else:
        # Add descriptions to entities from Wikipedia/Medline
        train_wiki_input_lst = list(
            map(lambda s: replace_entities(s, bio=bio_flag), train_input_lst)
        )
        test_wiki_input_lst = list(
            map(lambda s: replace_entities(s, bio=bio_flag), test_input_lst)
        )

    with open(f"misc/train_{input_dict['name']}_lst.pkl", "wb") as f:
        pickle.dump(train_wiki_input_lst, f)
    with open(f"misc/test_{input_dict['name']}_lst.pkl", "wb") as f:
        pickle.dump(test_wiki_input_lst, f)

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

for train_idx, test_idx, filter_type in [input_lst_all, input_lst_some]:
    train_label_lst_ = [[lst[i] for i in train_idx] for lst in train_label_lst]
    train_input_lst_ = [train_wiki_input_lst[i][0] for i in train_idx]
    test_label_lst_ = [[lst[i] for i in test_idx] for lst in test_label_lst]
    test_input_lst_ = [test_wiki_input_lst[i][0] for i in test_idx]

    # Write individual json for each row, first using one reference per input (for training)
    train_json, test_json = [], []
    for lst in train_label_lst_:
        train_json.extend(
            [{"input": a, "labels": [b]} for (a, b) in zip(train_input_lst_, lst)]
        )
    for lst in test_label_lst_:
        test_json.extend(
            [{"input": a, "labels": [b]} for (a, b) in zip(test_input_lst_, lst)]
        )
    output_json_indiv = {"train": train_json, "test": test_json}

    # Output the file
    write_json(
        output_json_indiv, f"data/{input_dict['name']}_context_{filter_type}.json"
    )

    # Then multiple references per input (for evaluation)
    train_json = [
        {"input": a, "labels": b}
        for (a, b) in zip(train_input_lst_, zip(*train_label_lst_))
    ]
    test_json = [
        {"input": a, "labels": b}
        for (a, b) in zip(test_input_lst_, zip(*test_label_lst_))
    ]
    output_json_mult = {"train": train_json, "test": test_json}

    # Output the file
    write_json(
        output_json_mult,
        f"data/{input_dict['name']}_context_{filter_type}_multiple.json",
    )
