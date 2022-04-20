"""
Download the Question NLI dataset from https://gluebenchmark.com/tasks
"""

import pandas as pd

qnli = pd.read_csv('data/qnli/raw/dev.tsv', sep='\t', error_bad_lines=False)
gold = []
for c in qnli['label']:
    if c == 'entailment':
        gold.append('entailment')
    elif c == 'not_entailment':
        gold.append('non-entailment')
    else:
        print('unrecognized label')
qnli['gold'] = gold

qnli = qnli.drop('label', axis=1)
qnli = qnli.rename(columns={'sentence': 'premise', 'question': 'hypothesis', 'index': 'pairID'})
qnli.to_json('data/qnli/dev.jsonl', lines=True, orient='records')