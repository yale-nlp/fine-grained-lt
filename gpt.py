import openai
import pandas as pd
import numpy as np
import nltk
from datasets import Dataset, DatasetDict, load_metric, load_dataset
import argparse
import os

os.environ["OPENAI_API_KEY"] = " "
openai.api_key = os.getenv("OPENAI_API_KEY")

DATASET_NAME = "turkcorpus"
dataset = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="test"
)["train"]
test_output = []
for text in dataset["input"]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that simplifies text.",
            },
            {"role": "user", "content": f'Simplify this text: "{text}"'},
        ],
        temperature=0,
        max_tokens=256,
    )

    simplified_sen = response["choices"][0]["message"]["content"]
    test_output.append(simplified_sen)
    # open file in write mode
with open(f"output/{DATASET_NAME}_gpt3.txt", "w") as fp:
    for item in test_output:
        fp.write("%s\n" % item)
    print("Done")
