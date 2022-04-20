"""
Download data from https://github.com/facebookresearch/anli and store in data/anli/raw
"""

import pandas as pd
from pathlib import Path

data_dir = Path('data/anli')
split = 'test'
label_map = {'c': 'contradiction', 'e': 'entailment', 'n': 'neutral'}
rounds = {}
for r in [1,2,3]:
    df = pd.read_json(data_dir / f'raw/R{r}/{split}.jsonl', lines=True)
    df = df.replace({'label': label_map})
    df = df.rename(columns={'label': 'gold', 'uid': 'pairID', 'context': 'premise'})
    df = df.drop(['model_label', 'emturk', 'reason', 'tag'], axis=1)
    df.to_json(data_dir / f'R{r}_{split}.jsonl', orient='records', lines=True)
    rounds[r] = df

pd.concat(rounds.values()).to_json(data_dir / f'{split}.jsonl', orient='records', lines=True)