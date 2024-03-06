# fine-grained-lt

Welcome to the Simplification Projects repository!

We aim to explore ways to use various training strategies to improve aspects of summarization and simplification.

## TLDRs
* We use unlikelihood learning and a modified decoding strategy to improve the simplicity/readability of models' outputs (<a href="https://aclanthology.org/2023.findings-emnlp.322">EMNLP 2023 Findings</a>). Check out a demo on <a href="https://huggingface.co/spaces/ljyflores/simplification-model-app">Streamlit</a>!
* We use variants of loss truncation to remove noisy examples from training, which reduces entity-level hallucination in model's outputs (<a href="https://openreview.net/forum?id=QFGsa3f-plp">EACL 2024</a>)

## Set-up
To get started, clone this repository and set up a `simplification` environment as follows:
```
# Set up the environment
conda create --name simplification python=3.8
conda activate simplification
pip install -r requirements.txt

# Install SciSpacy models
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.4.1/en_core_web_lg-3.4.1-py3-none-any.whl
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
```

We've moved the training losses into a separate package called <a href="https://github.com/ljyflores/loss-library">Loss Library</a> (help us build this!). 

The goal is to allow users to plug-and-play training losses from various research papers. It can be installed as follows:
```
# Install Loss Library
git clone https://github.com/ljyflores/loss-library.git
cd loss-library
pip install .
```

Then install the evaluation package `easse`
```
git clone https://github.com/feralvam/easse.git
cd easse
pip install -e .
```

We also set up a separate `simplification_questeval` environment to use QuestEval separately, as it conflicts with `simplification`.

```
# Set up the environment
conda create --name simplification_questeval python=3.9
conda activate simplification_questeval
pip install -r requirements_questeval.txt 
```

<b>OpenAI: </b> We also use the OpenAI API for various generation and evaluation tasks. To save your key (and use the evaluation script), first create a file called `openai_key` (no `.txt` extension!), then paste your key there! The script will look for this file and read the key from it. This key is also automatically excluded using the `.gitignore`, so we don't push our keys here online.

<b>Weights and Biases:</b> Before training, be sure to log in to <a href="https://wandb.ai/">wandb</a> (weights and biases), which helps us log experiments and keep track of their performance. To do this, set up a (free) account with wandb, then copy the API key! Back in the terminal, we can log in as follows:
```
wandb login <API_KEY>
```

## Data
Each dataset is stored using `json` files – each with two versions: `<dataset>.json` and `<dataset>_multiple.json`. 
* `<dataset>.json` is used for training
  * one example input is mapped to exactly one training label – so if there are 10 labels written for one input, the `<dataset>.json` will have 10 examples for this input
* `<dataset>_multiple.json` is used for evaluation
  *  one example input is mapped to all the training labels written for it – so if there are 10 labels written for one input, the `<dataset>_multiple.json` will still only have 1 example for it

So far, Cochrane, ASSET, and MedEasi are available from this repository. We are uploading the processed files for CNN and XSum to a repository (coming soon).

To use your own data, kindly ensure it is in the following format:
```
{
  "train": [{"input": <str>, "labels": [<str>, <str>, ..., <str>]}, {"input": <str>, "labels": [<str>, <str>, ..., <str>]}],
  "test":  [{"input": <str>, "labels": [<str>, <str>, ..., <str>]}, {"input": <str>, "labels": [<str>, <str>, ..., <str>]}]
}
```

## Training
| Parameter | Description | Values |
| --------- | ----------- | ------ |
| `dataset` | Dataset to use | `asset_full`, `cochrane_full`, `medeasi`, `cnn_full`, `xsum_full` |
| `model`   | Model to use | `bart`, `bart_xsum`, `flant5`, `flant5_base` |
| `checkpoint` | Checkpoint to use (optional) | Checkpoint path, passed into `AutoModel` |
| `loss_type` | Loss to use (optional, defaults to NLL) | `ul` (Unlikelihood Loss), `lt` (Loss Truncation), `max_lt` (Entity-Level Loss Truncation) (See <a href="https://github.com/ljyflores/loss-library">LossLibrary</a>) |
| `hyperparameter_tune` | Whether or not to do hyperparameter tuning | `True` or `False` |
| `predict_only` | Whether or not to just run prediction | `True` or `False` |
| `learning_rate` | Learning rate | Float |
| `epochs` | Number of epochs | Integer |
| `batch_size` | Batch size | Integer |
| `gradient_accumulation_steps` | Gradient accumulation steps | Integer |
| `scheduler` | Scheduler type | Either `linear` or `constant` |

We've set up a script that reads in dataset from the `data` folder and trains a model with the specified parameters.
It then outputs a textfile of the summaries generated by the model for the `test` set, and places them in the `output` folder as `<dataset_name>.txt`.
Here's the structure of the command to train a model:
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --dataset <dataset> --model <model> --loss_type <loss_type> --lr <lr> --epochs <num_epochs> --batch_size <batch_size> --gradient_accumulation_steps <grad_acc> 
```

For example,
```
CUDA_VISIBLE_DEVICES=7 python train.py --dataset asset --lr 5e-4 --epochs 10 --batch_size 8 --gradient_accumulation_steps 8 --model bart_xsum
```
To use unlikelihood loss, add the training parameter `--loss_type ul`, which will automatically add the readability penalty. To further add the consistency penalty, use either `ul_inp`, `ul_lab`, or `ul_inp_lab`, which will penalize the model for generating entities that are not present in either the input, label, or both.

## Decoding
To run the decoding script, we require a model checkpoint and a dataset to use. 

Parameters like the intervals at which BERTScore and FK are calculated and the number of beams can be set in the script.

```
CUDA_VISIBLE_DEVICES=<gpu_id> python decode.py --dataset <dataset> --model <model> --checkpoint <checkpoint>
```

## Evaluation

To run the evaluation script that compares the output `.txt` file to the reference summaries in the `.json` file, run this command:
```
CUDA_VISIBLE_DEVICES=<gpu_id> python eval.py --dataset <dataset> --preds_path <model predictions>
```

## Citing

If you found our work useful, kindly cite it for more people to learn about it! 
```
@inproceedings{flores-etal-2023-medical,
    title = "Medical Text Simplification: Optimizing for Readability with Unlikelihood Training and Reranked Beam Search Decoding",
    author = "Flores, Lorenzo Jaime  and Huang, Heyuan  and Shi, Kejian  and Chheang, Sophie  and Cohan, Arman",
    editor = "Bouamor, Houda  and Pino, Juan  and Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.322",
    doi = "10.18653/v1/2023.findings-emnlp.322",
    pages = "4859--4873",
}

@inproceedings{
    flores2024on,
    title = "On the Benefits of Fine-Grained Loss Truncation: A Case Study on Factuality in Summarization",
    author = "Flores, Lorenzo Jaime and Cohan, Arman",
    booktitle = "18th Conference of the European Chapter of the Association for Computational Linguistics",
    year = "2024",
    url = "https://openreview.net/forum?id=tv1NSTp0GE"
}
```
