import pandas as pd
import json
import re


def write_json(output_json, path):
    json_object = json.dumps(output_json, indent=4)
    with open(path, "w") as outfile:
        outfile.write(json_object)


def clean_spaces(x):
    return re.sub(" +", " ", x.strip())


def extract_data(df):
    return [
        {
            "input": clean_spaces(row["sentence"]),
            "labels": [clean_spaces(row["simplified_sentence"])],
            "vocab": list(
                filter(
                    lambda x: x != "",
                    list(map(clean_spaces, row["difficult_words"].split(","))),
                )
            ),
            "report_id": row["report_id"],
        }
        for i, row in df.copy().iterrows()
    ]


df = pd.read_excel("data/raw/radiology/annotations-sorted.xlsx")
df = (
    df[["report_id", "order", "sentence", "simplified_sentence", "difficult_words"]]
    .sort_values(["report_id", "order"])
    .reset_index(drop=True)
)

# Full means each training point is the full report
df_full = df.copy()
df_full["sentence"] = df_full["sentence"].fillna("")
df_full["simplified_sentence"] = df_full["simplified_sentence"].fillna("")
df_full["difficult_words"] = df_full["difficult_words"].fillna("")
df_full = (
    df_full.groupby("report_id")
    .aggregate(
        {
            "sentence": lambda x: " ".join(x),
            "simplified_sentence": lambda x: " ".join(x),
            "difficult_words": lambda x: " ".join(x),
        }
    )
    .reset_index()
)
# Indiv means each training point is on the sentence level
df_indiv = df.copy()
df_indiv = df_indiv.loc[
    ~df_indiv["sentence"].isnull() & ~df_indiv["simplified_sentence"].isnull()
].reset_index(drop=True)
df_indiv["difficult_words"] = (
    df_indiv["difficult_words"].fillna("").reset_index(drop=True)
)

# Train: Report 0 to 1890
# Test:  Report 1891 to 2364
df_full_train = df_full.loc[df_full.report_id <= 1890]
df_full_test = df_full.loc[df_full.report_id > 1890]

df_indiv_train = df_indiv.loc[df_indiv.report_id <= 1890]
df_indiv_test = df_indiv.loc[df_indiv.report_id > 1890]

# Extract the line items
output_json_full = {
    "train": extract_data(df_full_train),
    "test": extract_data(df_full_test),
}
output_json_indiv = {
    "train": extract_data(df_indiv_train),
    "test": extract_data(df_indiv_test),
}

# Filter out nulls from full
output_json_full["train"] = list(
    filter(lambda d: d["labels"][0] != "", output_json_full["train"])
)
output_json_full["test"] = list(
    filter(lambda d: d["labels"][0] != "", output_json_full["test"])
)

output_json_indiv["train"] = list(
    filter(lambda d: d["labels"][0] != "", output_json_indiv["train"])
)
output_json_indiv["test"] = list(
    filter(lambda d: d["labels"][0] != "", output_json_indiv["test"])
)

# Output the file
write_json(output_json_full, f"data/radiology_full.json")
write_json(output_json_full, f"data/radiology_full_multiple.json")
write_json(output_json_indiv, f"data/radiology_indiv.json")
write_json(output_json_indiv, f"data/radiology_indiv_multiple.json")
