"""
heuristic-based filtering
"""

import pandas as pd
import numpy as np
import click
from utils.utils import strip_punctuation_and_casing
import os
from pathlib import Path
from tqdm import tqdm

forbidden_phrases = ['pair of sentences', 'previous examples', 'same relationship as', 'The speaker']
dangling_punctuation = ['"', "'", '(', ')']


def clean(sentence):
    sentence = sentence.strip()
    if sentence.count('"') == 2 and sentence[0] == '"' and sentence[-1] == '"':
        sentence = sentence[1:-1]
    if sentence[0] == '(' and sentence[-1] == ')' and (sentence.count('(') + sentence.count(')')) == 2:
        sentence = sentence[1:-1]
    for p in dangling_punctuation:
        if sentence[0] == p and sentence.count(p) == 1:
            sentence = sentence[1:]
        elif sentence[-1] == p and sentence.count(p) == 1:
            sentence = sentence[:-1]
    return sentence


@click.command()
@click.option('--data_file', type=str, help='jsonl file of examples to clean (should be named examples.jsonl)')
@click.option('--original_data_file', type=str, default='data/mnli/train.jsonl', help='jsonl of examples from original dataset')
def main(data_file: str, original_data_file: str):
    dataset_df = pd.read_json(data_file, lines=True)
    dataset_df = dataset_df.reset_index().rename({'index': 'id'}, axis=1)
    output_dir = Path(os.path.dirname(data_file))
    print(f'Output directory set to {output_dir}')
    mnli = pd.read_json(original_data_file, lines=True, orient='records')

    discards = {
        'short': [],                 # premise or hypothesis has fewer than 2 characters
        'copied_premise': [],        # premise == hypothesis
        'copied_nn': [],             # examples copied nearest neighbor
        'forbidden_phrase': [],      # examples contain phrase from instructions
    }

    to_drop = []
    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df.index)):
        if row['premise'] is None or row['hypothesis'] is None:
            discards['short'].append(idx)
            continue
        premise, hypothesis = row['premise'].strip(), row['hypothesis'].strip()
        # 1. filter examples where premise or hypothesis is too short
        if min(len(premise), len(hypothesis)) < 5:
            discards['short'].append(idx)
            continue
        # 2. filter examples where hypothesis == premise, ignoring punctuation and casing
        if strip_punctuation_and_casing(premise) == strip_punctuation_and_casing(hypothesis):
            discards['copied_premise'].append(idx)
            continue
        # 3. filter examples that contain a forbidden phrase
        if np.any([x in premise + hypothesis for x in forbidden_phrases]):
            discards['forbidden_phrase'].append(idx)
            continue
        # 4. filter examples where the example copies an in-context example
        copied_nn = False
        for nn in row['nearest_neighbors']:
            nn_premise = mnli.loc[nn]['premise'].strip()
            nn_hypothesis = mnli.loc[nn]['hypothesis'].strip()
            if premise == nn_premise and hypothesis == nn_hypothesis:
                copied_nn = True
                break
        if copied_nn:
            discards['copied_nn'].append(idx)
            continue
        # clean examples to strip whitespaces and weird punctuation
        dataset_df.at[idx, 'premise'] = clean(premise)
        dataset_df.at[idx, 'hypothesis'] = clean(hypothesis)
    
    to_drop = [idx for sublist in discards.values() for idx in sublist]
    dataset_df = dataset_df.drop(to_drop)

    print(f'Filtered data written to {data_file}')
    dataset_df.to_json(output_dir / f'filtered_examples.jsonl', orient='records', lines=True)
    print(f'Filtered data written to {output_dir}/filtered_examples.jsonl')
    
    for k, v in discards.items():
        print(f'{k}\t\t{len(v)}')


if __name__ == "__main__":
    main()