"""
anonymize data by mapping AMT WorkerIds to aliases
"""

import pandas as pd
import json
import random
import string

df = pd.read_json('annotation/all_batches/processed_results.jsonl', lines=True)

# create a dictionary from AMT WorkerIDs to aliases
worker_id_to_alias = {w: ''.join(random.choice(string.ascii_uppercase) for i in range(3)) for w in set(df['WorkerId'].tolist())}
assert len(set(worker_id_to_alias.values())) == len(set(worker_id_to_alias.keys()))

df['WorkerId'] = [worker_id_to_alias[w] for w in df['WorkerId']]
df.drop(['TimeOnPageInSeconds', 'HITId'], axis=1, inplace=True)

df.to_json('annotation/all_batches/anonymized_annotations.jsonl', lines=True, orient='records')