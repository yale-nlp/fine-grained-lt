import openai
import pandas as pd
import numpy as np
import nltk
from datasets import Dataset, DatasetDict, load_metric, load_dataset
import argparse
import os
import re

os.environ["OPENAI_API_KEY"] = " "
openai.api_key = os.getenv("OPENAI_API_KEY")


# DATASET_NAME = "turkcorpus"
# DATASET_NAME = "radiology_indiv"
DATASET_NAME = "cochrane"
# DATASET_NAME = "asset"
dataset = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="test"
)["train"]
# test_output = []
for text in dataset["input"]:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that simplifies text.",
            },
            {
                "role": "user",
                "content": f"Simplify this text: {text}",
                # "content": f"Simplify this text for a fourth grade or lower student to understand: {text}",
            },
        ],
        temperature=0,
        max_tokens=128,
    )

    simplified_sen = response["choices"][0]["message"]["content"]

    # Remove newlines, replace multiple spaces with one
    simplified_sen = simplified_sen.replace("\n", " ")
    simplified_sen = re.sub(" +", " ", simplified_sen)

    # test_output.append(simplified_sen)
    # open file in write mode
    with open(f"output/{DATASET_NAME}_gpt4_basic_128.txt", "a") as fp:
        # for item in test_output:
        fp.write("%s\n" % simplified_sen)

print(f"{DATASET_NAME} Done")
