"""
The dataset was shared with us by authors
"""
import pandas as pd

# filtering for fluency recommended by authors
df = tailor_df.loc[tailor_df['perturb_strategy'] == 'swap_core']
thres = df['perplex'].quantile(q=0.5)
df = df.loc[df['perplex'] < thres]
df = df.loc[~df['clean_generated_sent'].str.contains('sanatate')]

# dataframe stuff
df = df.reset_index().drop('index', axis=1).reset_index()
label_map = {True: 'entailment', False: 'neutral'}
df = df.replace({'bool_preserve_meaning': label_map})
df = df.rename(columns={'index': 'id', 'sentence2':'premise', 'clean_generated_sent': 'hypothesis', 'bool_preserve_meaning': 'gold'})
df.drop(columns=['annotator_labels', 'captionID', 'prompt', 'sentence1_parse', 'sentence1_binary_parse', 'sentence2_parse', 'sentence2_binary_parse', 'perplex', 'perturb_strategy', 'generated_sent', 'sent_modified', 'which_sent_modified', 'data_idx', 'sentence1', 'gold_label'], inplace=True)

# data cleaning recommended by authors
for i, row in df.iterrows():
    premise, hypothesis = row['premise'], row['hypothesis']
    premise = premise.replace(' , ', ', ')          # remove additional spacing before commas
    hypothesis = hypothesis.replace(' , ', ', ')
    if premise[-2:] == ' .':                        # remove additional space before periods
        premise = premise[:-2] + '.'
    if hypothesis[-2:] == ' .':
        hypothesis = hypothesis[:-2] + '.'
    df.at[i, 'premise'] = premise
    df.at[i, 'hypothesis'] = hypothesis

df.to_json(data_dir / 'tailor.jsonl', orient='records', lines=True)