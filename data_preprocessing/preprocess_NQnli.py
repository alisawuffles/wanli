"""
Download the data from https://github.com/jifan-chen/QA-Verification-Via-NLI and store in data/na-nli/raw
"""

import pandas as pd
from pathlib import Path

data_dir = Path('data/nq-nli')
split = 'train'
nq_nli = pd.read_json(data_dir / f'raw/nq-nli-{split}.jsonl', lines=True)
label_map = {True: 'entailment', False: 'non-entailment'}
nq_nli = nq_nli.replace({'is_correct': label_map})

nq_nli = nq_nli.rename(columns={'example_id': 'pairID', 'decontext_answer_sent_text': 'premise', 'question_statement_text': 'hypothesis', 'is_correct': 'gold'})
cols_drop = ['title_text', 'paragraph_text', 'answer_sent_text', 'question_text', 'has_gold', 'answer_text', 'decontextualized_sentence', 'category', 'answer_score', 'answer_scores', 'kamath_score', 'f1', 'gold_answers', 'f1_score']
for col in cols_drop:
    if col in nq_nli.columns:
        nq_nli = nq_nli.drop(col, axis=1)
nq_nli.to_json(f'data/nq-nli/{split}.jsonl', orient='records', lines=True)