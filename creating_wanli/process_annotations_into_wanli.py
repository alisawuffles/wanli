"""
use postprocessing logic to transofrm annotation results into NLI examples
"""

from utils.utils import get_genre, example_revised, only_punctuation_revised
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    all_batch_dir = Path(f'annotation/all_batches')
    results_df = pd.read_json(all_batch_dir / f'processed_results.jsonl', lines=True)

    deduplicated_rows = []
    for example_id in tqdm(set(results_df.id)):
        sub_df = results_df.loc[results_df['id'] == example_id].sample(frac=1)
        row1 = sub_df.iloc[0].to_dict()
        row2 = sub_df.iloc[1].to_dict()
        # if either worker wants to discard, then discard
        if row1['gold'] == 'discard':
            deduplicated_rows.append(row1)
        elif row2['gold'] == 'discard':
            deduplicated_rows.append(row2)
        # if both workers revised, then keep a revised row, with no preference toward which revision
        elif example_revised(row1) and example_revised(row2):
            deduplicated_rows.append(row1)
        # if only one worker revised but it's only a punctuation revision, then keep the revised example
        elif example_revised(row1) and only_punctuation_revised(row1):
            deduplicated_rows.append(row1)
        elif example_revised(row2) and only_punctuation_revised(row2):
            deduplicated_rows.append(row2)
        # otherwise if only one worker revised, then keep the non-revised example
        elif example_revised(row1):
            deduplicated_rows.append(row2)
        elif example_revised(row2):
            deduplicated_rows.append(row1)
        # if neither worker revised or discarded, then randomly sample an annotation
        else:
            deduplicated_rows.append(row1)

    # reformat dataframe
    df = pd.DataFrame(deduplicated_rows)
    assert set(df.id) == set(results_df.id)
    print(f'Total number of annotated examples: {len(df.index)}')
    df['genre'] = df.apply(get_genre, axis=1)
    df = df.drop(['premise', 'hypothesis'], axis=1)
    df = df.rename(columns={'revised_premise': 'premise', 'revised_hypothesis': 'hypothesis'})
    df['pairID'] = [str(ns[0]) for ns in df['nearest_neighbors'].tolist()]
    df = df.drop(['WorkerId', 'HITId', 'TimeOnPageInSeconds', 'revised', 'nearest_neighbors', 'label'], axis=1)

    # separate into different dataframes
    labeled_df = df[df.gold != 'discard'].sample(frac=1)        # WANLI
    print(f'Number of labeled examples: {len(labeled_df.index)}')
    revised_df = labeled_df[labeled_df.genre == 'generated_revised'].sample(frac=1)     # subset of WANLI that was revised
    print(f'Number of revised examples: {len(revised_df.index)}')
    discard_df = df[df.gold == 'discard'].sample(frac=1)        # discarded examples (not in WANLI)
    print(f'Number of discarded examples: {len(discard_df.index)}')

    print(f"Percentage of WANLI examples that were revised: {np.round(labeled_df.genre.value_counts()['generated_revised']/len(labeled_df.index)*100,4)}%")
    print(f"Percentage of examples discarded: {np.round(len(discard_df.index)/len(df.index)*100,4)}%")

    labeled_df.to_json(all_batch_dir / f'labeled_examples.jsonl', orient='records', lines=True)
    revised_df.to_json(all_batch_dir / f'revised_examples.jsonl', orient='records', lines=True)
    discard_df.to_json(all_batch_dir / f'discarded_examples.jsonl', orient='records', lines=True)


if __name__ == "__main__":
    main()