"""
Download MNLI data from https://cims.nyu.edu/~sbowman/multinli/ and store in data/mnli/raw
"""
import pandas as pd

for split in ['train', 'dev_matched', 'dev_mismatched']:
    mnli = pd.read_json(f'data/mnli/raw/{split}.jsonl', lines=True, orient='records')
    mnli.reset_index(level=0, inplace=True)
    mnli.drop(columns=['promptID', 'annotator_labels', 'sentence1_parse', 'sentence1_binary_parse', 'sentence2_parse', 'sentence2_binary_parse'], inplace=True)
    mnli = mnli.rename(columns={
        'sentence1': 'premise', 
        'sentence2': 'hypothesis', 
        'index': 'id', 
        'gold_label': 'gold'
    })
    mnli.to_json(f'data/mnli/{split}.jsonl', orient='records', lines=True)