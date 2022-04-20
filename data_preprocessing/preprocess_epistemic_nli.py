"""
Download data from https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/epistemic_reasoning
and store in data/epistemic_reasoning/raw
"""

import pandas as pd
import json

with open('data/epistemic_reasoning/raw/task.json', 'r') as fin:
    data = fin.read()

data = json.loads(data)

nli_examples = []
examples = data['examples']
for row in examples:
    s = row['input'].split('Premise: ')[1]
    premise, hypothesis = s.split(' Hypothesis: ')
    gold = 'entailment' if row['target_scores']['entailment'] == 1 else 'non-entailment'
    ex = {
        'premise': premise,
        'hypothesis': hypothesis,
        'gold': gold
    }
    nli_examples.append(ex)

df = pd.DataFrame(nli_examples)
df = df.reset_index().drop('index', axis=1).reset_index().rename(columns={'index': 'id'})

df.to_json('data/epistemic_reasoning/test.jsonl', orient='records', lines=True)