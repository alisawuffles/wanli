import click
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os, glob
import pandas as pd
import torch
from utils.utils import ensure_dir
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path


def evaluate(
    model_path: Path,
    evaluation_file: str, 
    dynamics_dir_name: str='training_dynamics', 
):
    """
    model_path: path to trained NLI model with saved checkpoints
    evaluation_file: path to data whose training dynamics are to be evaluated

    for each checkpoint, create a jsonl file of predictions
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device('cuda')
    evaluation_df = pd.read_json(evaluation_file, lines=True)
    premise_key, hypothesis_key, label_key = 'premise', 'hypothesis', 'gold'
    has_gold = label_key in evaluation_df.columns
    output_dir = model_path / dynamics_dir_name
    ensure_dir(output_dir)
    
    checkpoints = list(
        int(os.path.dirname(c).split('-')[-1]) for c in 
            glob.glob(str(model_path / "checkpoint-*/pytorch_model.bin"), recursive=True)
    )
    checkpoints = [model_path / f'checkpoint-{c}' for c in sorted(checkpoints)]
    epochs = range(len(checkpoints))
    print(f'Evaluating the following checkpoints: {checkpoints}')
    
    for epoch, checkpoint in zip(epochs, checkpoints):
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        model.to(device)
        ids, logits = [], []
        if has_gold:
            labels = []
        with torch.no_grad():
            for i, row in tqdm(evaluation_df.iterrows(), desc=f'Epoch {epoch}', total=len(evaluation_df.index)):
                premise, hypothesis, label = row[premise_key], row[hypothesis_key], row[label_key]
                inputs = tokenizer(premise, hypothesis, return_tensors='pt', max_length=128, truncation=True).to(device)
                l = model(**inputs).logits
                ids.append(row['id'])
                logits.append(l.squeeze(0))
                if has_gold:
                    labels.append(label)

        logits = torch.stack(logits)
        logits = list(logits.cpu().detach().numpy())

        epoch_predictions = {'id': ids, f'logits_epoch_{epoch}': logits}
        if has_gold:
            epoch_predictions[label_key] = labels
        dynamics_df = pd.DataFrame(epoch_predictions)
        epoch_filename = output_dir / f'dynamics_epoch_{epoch}.jsonl'
        dynamics_df.to_json(epoch_filename, orient='records', lines=True)
        print(f'Epoch {epoch} dynamics logged to {epoch_filename}')


def read_training_dynamics(dynamics_dir: Path):
    """
    model_dir: path to logged training dynamics
    merges stats across epochs
    returns a dictionary from example ID to the list of logits across epochs and (if there is one) its gold label
    """
    train_dynamics = {}
    num_epochs = len([f for f in os.listdir(dynamics_dir) if (os.path.isfile(dynamics_dir / f) and f.startswith('dynamics_epoch'))])

    print(f"Reading {num_epochs} files from {dynamics_dir} ...")
    for epoch_num in tqdm(range(num_epochs)):
        epoch_file = dynamics_dir / f"dynamics_epoch_{epoch_num}.jsonl"
        assert os.path.exists(epoch_file)

        dynamics_df = pd.read_json(epoch_file, lines=True)
        has_gold = 'gold' in dynamics_df.columns
        for i, row in dynamics_df.iterrows():
            guid = row['id']
            if guid not in train_dynamics:
                assert epoch_num == 0
                train_dynamics[guid] = {"logits": []}
                if has_gold:
                    train_dynamics[guid]['gold'] = row['gold']
            train_dynamics[guid]["logits"].append(row[f"logits_epoch_{epoch_num}"])

    return train_dynamics


def compute_td_metrics(train_dynamics, label2id):
    has_gold = 'gold' in list(train_dynamics.items())[0][1]
    max_variability_ = {}

    if has_gold:
        variability_, confidence_, correctness_ = {}, {}, {}

    num_tot_epochs = np.max([len(record['logits']) for record in train_dynamics.values()])
    print(f"Computing training dynamics across {num_tot_epochs} epochs")
    
    for guid in tqdm(train_dynamics.keys()):
        if has_gold:
            correctness_trend = []
            true_probs_trend = []
        probs_per_label = [[],[],[]]

        record = train_dynamics[guid]
        # skip examples that we do not have training dynamics for all epochs for
        if len(record['logits']) < num_tot_epochs:
            continue
        for i, epoch_logits in enumerate(record["logits"]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            for l in range(3):
                probs_per_label[l].append(float(probs[l]))

            if has_gold:
                true_class_prob = float(probs[label2id[record["gold"]]])
                true_probs_trend.append(true_class_prob)
                
                prediction = np.argmax(epoch_logits)
                is_correct = prediction == label2id[record["gold"]]
                correctness_trend.append(is_correct)

        max_variability_[guid] = np.max([np.std(probs_per_label[l]) for l in range(3)])

        if has_gold:
            correctness_[guid] = sum(correctness_trend)
            confidence_[guid] = np.mean(true_probs_trend)
            variability_[guid] = np.std(true_probs_trend)

    column_names = ['guid', 'max_variability']
    if has_gold:
        column_names.extend(['correctness', 'confidence', 'variability'])
    
    if not has_gold:
        metrics_df = pd.DataFrame([
            [guid, max_variability_[guid]] 
            for i, guid in enumerate(max_variability_)], columns=column_names)
    else:
        metrics_df = pd.DataFrame([
            [guid, max_variability_[guid], correctness_[guid], confidence_[guid], variability_[guid]] 
            for i, guid in enumerate(max_variability_)], columns=column_names)

    return metrics_df


@click.command()
@click.option('--model_path', type=str)
@click.option('--evaluation_file', type=str)
@click.option('--dynamics_dir_name', type=str, default='training_dynamics')
@click.option('--overwrite_train_dy/--reuse_train_dy', default=False)
def main(
    model_path: str, 
    evaluation_file: str, 
    dynamics_dir_name: str,
    overwrite_train_dy: bool
):  
    # evaluate each model checkpoint on the evaluation data, and store predictions 
    # in dynamics_dir_name, one for each epoch
    model_path = Path(model_path)
    model_config = json.load(open(model_path / 'config.json'))
    label2id = model_config['label2id']
    if overwrite_train_dy or not os.path.exists(model_path / f'{dynamics_dir_name}/dynamics_epoch_0.jsonl'):
        evaluate(model_path, evaluation_file, dynamics_dir_name=dynamics_dir_name)
    
    # read the training dynamics and merge stats across epochs
    dynamics_dir = model_path / dynamics_dir_name
    train_dynamics = read_training_dynamics(dynamics_dir)
    
    # compute metrics like max variability
    metrics_df = compute_td_metrics(train_dynamics, label2id=label2id)
    metrics_df.to_json(dynamics_dir / 'td_metrics.jsonl', orient='records', lines=True)


if __name__ == '__main__':
    main()
