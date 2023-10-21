import json


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

for input_dict in [ASSET_FILES, COCH_FILES, RADR_FILES, TURK_FILES]:
    # Read files
    train_input_lst = read_json(input_dict["train_input"])
    train_label_lst = [read_json(f) for f in input_dict["train_labels"]]
    test_input_lst = read_json(input_dict["test_input"])
    test_label_lst = [read_json(f) for f in input_dict["test_labels"]]

    # Write individual json for each row, first using one reference per input (for training)
    train_json, test_json = [], []
    for lst in train_label_lst:
        train_json.extend(
            [{"input": a, "labels": [b]} for (a, b) in zip(train_input_lst, lst)]
        )
    for lst in test_label_lst:
        test_json.extend(
            [{"input": a, "labels": [b]} for (a, b) in zip(test_input_lst, lst)]
        )
    output_json = {"train": train_json, "test": test_json}

    # Output the file
    write_json(output_json, f"data/{input_dict['name']}.json")

    # Then multiple references per input (for evaluation)
    train_json = [
        {"input": a, "labels": b}
        for (a, b) in zip(train_input_lst, zip(*train_label_lst))
    ]
    test_json = [
        {"input": a, "labels": b}
        for (a, b) in zip(test_input_lst, zip(*test_label_lst))
    ]
    output_json = {"train": train_json, "test": test_json}

    # Output the file
    write_json(output_json, f"data/{input_dict['name']}_multiple.json")
