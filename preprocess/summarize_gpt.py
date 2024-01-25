import openai
from datasets import Dataset, DatasetDict, load_metric, load_dataset
import os
import re
import time

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.getenv("OPENAI_API_KEY")


# DATASET_NAME = "turkcorpus_context_all"
# DATASET_NAME = "radiology_indiv_context_some"
# DATASET_NAME = "cochrane_context_some"
# DATASET_NAME = "asset_context_all"
# DATASET_NAME = "radiology_full"
# DATASET_NAME = "tico19"
DATASET_NAME = "medeasi"

dataset = load_dataset(
    "json", data_files=f"data/{DATASET_NAME}_multiple.json", field="test"
)["train"]

for idx, text in enumerate(dataset["input"]):
    time.sleep(0.01)
    try:
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

        # open file in write mode
        # with open(f"output/{DATASET_NAME}_gpt4_basic_128.txt", "a") as fp:
        #     # for item in test_output:
        #     fp.write("%s\n" % simplified_sen)
        print(idx, simplified_sen)

    except Exception as e:
        print(f"Error on {idx}: {e}")
        print(text)
        # with open(f"output/{DATASET_NAME}_gpt4_basic_128.txt", "a") as fp:
        #     # for item in test_output:
        #     fp.write("%s\n" % "")
        # continue
    
print(f"{DATASET_NAME} Done")

# openai.error.InvalidRequestError: This model's maximum context length is 8192 tokens. However, your messages resulted in 8534 tokens. Please reduce the length of the messages.
