"""
Download data from https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md
and store in data/fever-nli/raw
"""
import pandas as pd
from pathlib import Path

data_dir = Path('data/fever-nli')
for split in ['train', 'dev', 'test']:
    df = pd.read_json(data_dir / f'raw/{split}_fitems.jsonl', lines=True)
    if split == 'dev':  # retrieve hidden labels in dev set
        df = df.drop(['verifiable', 'label'], axis=1)
        fever_df = pd.read_json(data_dir / f'fever_raw/shared_task_{split}.jsonl', lines=True).drop(['claim', 'evidence'], axis=1)
        df = df.merge(fever_df, how='inner', left_on='cid', right_on='id', validate='one_to_one')
        df.drop('id', axis=1, inplace=True)
    label_map = {'SUPPORTS': 'entailment', 'REFUTES': 'contradiction', 'NOT ENOUGH INFO': 'neutral'}
    df = df.replace({'label': label_map})
    df.reset_index(level=0, inplace=True)
    df = df.rename(columns={'index': 'id', 'cid':'pairID', 'query': 'hypothesis', 'context': 'premise', 'label': 'gold'})
    df = df.drop(['fid', 'verifiable'], axis=1)
    df.to_json(data_dir / f'{split}.jsonl', lines=True, orient='records')
