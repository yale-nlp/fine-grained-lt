# simplification-project

Welcome to the simplification project repository! 
We aim to explore ways to train language models to simplify radiology reports to make them more accessible to laypeople.

### Set-up
To get started, clone this repository and set up a `simplification` environment as follows:
```
# Clone the repo
git clone https://github.com/ljyflores/simplification-project
cd simplification-project

# Set up the environment
conda create --name simplification python=3.8
conda activate simplification
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install

```

### Data
Describe data format here

### Training
We've set up a script that takes in data from the `data` folder and trains a model with certain parameters.
Here's the structure of the command to train a model:
```
CUDA_VISIBLE_DEVICES=<gpu_id> WANDB_PROJECT=<wandb_project_name> nohup python train.py --dataset <dataset> --lr <lr> --epochs <num_epochs> --batch_size <batch_size> --gradient_accumulation_steps <grad_acc> --model <model> --weight_decay <weight_decay> >> <log_file_name>
```

For example,
```
CUDA_VISIBLE_DEVICES=7 WANDB_PROJECT=asset_flant5 nohup python train.py --dataset asset --lr 5e-4 --epochs 10 --batch_size 8 --gradient_accumulation_steps 8 --model flant5_base --weight_decay 0.05 >> nohup_asset_flant5_train.out
```

These are some notes on each of the parameters: 
* `dataset`: One of `asset`, `asset_context_all`, `cochrane`, `cochrane_context_all`, `turkcorpus`, `turkcorpus_context_all`, `radiology_indiv`, `radiology_indiv_context_all`
* `model`: One of `bart`, `flant5` (FLAN Large), `flant5_base`
* `weight_decay`: Similar to dropout (0 is none)
* `batch_size` and `gradient_accumulation_steps`: Actual batch size is the product of `batch_size` and `gradient_accumulation_steps`, `batch_size` controls how many samples fit on the model, while `gradient_accumulation_steps` is the number of steps taken before backpropagating

### Parameters
| `model`       | `dataset`                              | `lr` | `num_epochs` | `batch_size` | `gradient_accumulation_steps` | `weight_decay` |
| ------------- | -------------------------------------- | ---- | ------------ | ------------ | ----------------------------- | -------------- |
| `flant5_base` | `asset`/`radiology_indiv`/`turkcorpus` | 5e-4 | 10           | 8            | 8                             | 0.05           |
| `flant5_base` | `cochrane`                             | 5e-4 | 10           | 2            | 32                            | 0.05           |
| `bart`        | `asset`/`radiology_indiv`/`turkcorpus` | 1e-5 | 10           | 8            | 8                             | 0.01           |
| `bart`        | `cochrane`                             | 1e-5 | 10           | 2            | 16                            | 0.01           |
