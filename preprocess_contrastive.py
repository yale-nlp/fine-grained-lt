import argparse
import random
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--kg", required=True, type=str)
parser.add_argument("--loss_type", required=False, type=str, default="cs")
args = parser.parse_args()

# Load knowledge graph definitions
PATH = f"/home/lily/lyf6/Simplification-Project/data/{args.kg}"
KG_NAME = args.kg
filenames = os.listdir(PATH)

terms = [i[:-4].replace("_", " ") for i in filenames]
descs = [" ".join([line.strip() for line in open(f"{PATH}/{f}")]) for f in filenames]

if args.loss_type == "mse_minimize":
    df = pd.DataFrame(
        {
            "terms": terms,
            "definitions": descs,
        }
    )
else:
    # Create negative examples
    num_terms = len(terms)
    desc_lst = []

    k = 3
    shift_idxs = random.sample(range(1, num_terms), k=k)

    for shift in shift_idxs:
        desc_lst.extend(descs[shift:] + descs[:shift])

    if args.loss_type == "cs":
        # Create dataframe, dataset, and dataloader
        df = pd.DataFrame(
            {
                "terms": terms * (k + 1),
                "definitions": descs + desc_lst,
                "labels": [1.0] * (num_terms) + [0.0] * (num_terms * k),
            }
        )
    elif args.loss_type == "mse_contrastive":
        df = pd.DataFrame(
            {"terms": terms * k, "definitions": descs * k, "contrast": desc_lst}
        )
    else:
        assert False

df = df.dropna().reset_index(drop=True)
df.to_csv(f"data/contrastive/{args.kg}_{args.loss_type}.csv", index=False)
