"""
Download the HANS dataset from https://github.com/hansanon/hans and store in data/hans/raw
"""
import pandas as pd

hans = pd.read_csv('data/hans/raw/heuristics_evaluation_set.txt', sep='\t')
hans = hans.drop(['sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse'], axis=1)
hans = hans.rename(columns={
    'sentence1': 'premise', 
    'sentence2': 'hypothesis', 
    'index': 'id', 
    'gold_label': 'gold'
})

hans.to_json('data/hans/hans.jsonl', orient='records', lines=True)