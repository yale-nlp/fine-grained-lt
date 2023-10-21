import json
import random
import pandas as pd

report_list = json.load(open("data/radiology_indiv_multiple.json"))["train"]
idxs = random.sample(list(range(1891)), k=50)
report_list = list(filter(lambda d: d["report_id"] in idxs, report_list))

df = pd.DataFrame.from_records(report_list)

df = df.drop("labels", axis=1)
df.to_csv("data/raw/radiology/annotation_task_1.csv", index=False)
