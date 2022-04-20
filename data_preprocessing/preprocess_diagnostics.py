"""
Download the Broadcoverage Diagnostics dataset from https://super.gluebenchmark.com/tasks and store in data/diagnostics/raw
"""

import pandas as pd

diagnostics = pd.read_csv('data/diagnostics/raw/diagnostic-full.tsv', sep='\t')
diagnostics = diagnostics.rename(columns={
    'Premise': 'premise', 
    'Hypothesis': 'hypothesis', 
    'index': 'id', 
    'Label': 'gold'
})

diagnostics.to_json('data/diagnostics/diagnostics.jsonl', orient='records', lines=True)