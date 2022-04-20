"""
Download Winograd NLI from https://gluebenchmark.com/tasks and store in data/wnli/raw
"""
import pandas as pd
from pathlib import Path

data_dir = Path('data/wnli')
split = 'train'
wnli = pd.read_csv(data_dir / f'raw/{split}.tsv', sep='\t')
label_map = {0: 'non-entailment', 1: 'entailment'}
wnli = wnli.replace({'label': label_map})
wnli = wnli.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis', 'index': 'pairID', 'label': 'gold'})

wnli.to_json(data_dir / f'{split}.jsonl', orient='records', lines=True)