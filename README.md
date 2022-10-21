# WANLI
This repository contains the code and data for [WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation](https://arxiv.org/abs/2201.05955).

WANLI (**W**orker-**A**I Collaboration for **NLI**) is a collection of 108K English sentence pairs for the task of natural language inference (NLI).
Each example is created by first identifying a "pocket" of examples in [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) that share a challenging reasoning pattern, then instructing GPT-3 to write a new example with the same pattern.
The set of generated examples are automatically filtered to contain those most likely to aid model training, and finally labeled and optionally revised by human annotators.
In this way, WANLI represents a new approach to dataset creation that combines the generative strength of language models and evaluative strength of humans.

## Model
You can download a WANLI-trained RoBERTa-large model from HuggingFace models [here](https://huggingface.co/alisawuffles/roberta-large-wanli). This also includes a small demo!

## Data
Download the WANLI dataset [here](https://allenai.org/data/wanli)! 

Other NLI datasets used in this work (including MultiNLI and the out-of-domain evaluation sets) can be found in this Google Drive [folder](https://drive.google.com/drive/u/0/folders/190bQgz1Mu8Do_0KOu6_NN84z9vai9nkj).

## Pipeline
Here are the steps to replicate the process of creating WANLI. Recall that the prerequisites of this pipeline are an existing dataset (we use MultiNLI) and a task model trained on this dataset (we finetune RoBERTa-large). The relevant scripts can be found in the `scripts/` folder.

1. Train RoBERTa-large on MultiNLI with `classification/run_nli.py`. The MultiNLI data is stored in `data/mnli/`. The model will be saved as `models/roberta-large-mnli`.
2. Retroactively compute the training dynamics for each example in the training set, using the saved checkpoints from the trained model, with `cartography/compute_training_dynamics.py`. The training dynamics will be stored inside the model directory. These statistics are used to collect the seed dataset via the most ambiguous p% of the training set.
3. In order to retrieve nearest neighbors for each seed example, we will pre-compute CLS token embeddings for all MultiNLI examples relative to the trained model. Use `representations/embed_examples.py` to produce a numpy file called `mnli.npy` inside the model directory.
4. Use `pipeline.py` to generate examples stored as `generated_data/examples.jsonl`! The pipeline uses the ambiguous seed examples found in step (2) and nearest neighbors found via the pre-computed embeddings from step (3), in order to generate examples with challenging reasoning patterns. For this step, you will need access to the GPT-3 API.
5. Heuristically filter the generated examples with `filtering/filter.py` to get `generated_data/filtered_examples.jsonl`.
6. Now we filter based on the estimated max variability, in order to keep examples most likely to aid model training. To do this, estimate the "training dynamics" of the generated data with respect to our trained task model, using `cartography/compute_train_dy_metrics.py`. Then, filter the dataset to keep examples with the highest estimated max variability using `filtering/keep_ambiguous.py` to create the final unlabeled dataset called `generated_data/ambiguous_examples.jsonl`. 
7. Recruit humans to annotate examples in the final data file! Use the processing scripts in `creating_wanli/` to process AMT batch results and postprocess them into NLI examples.

## Citation
```
@misc{liu-etal-2022-wanli,
    title = "WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation",
    author = "Liu, Alisa  and
      Swayamdipta, Swabha  and
      Smith, Noah A.  and
      Choi, Yejin",
    month = jan,
    year = "2022",
    url = "https://arxiv.org/pdf/2201.05955",
}
```
