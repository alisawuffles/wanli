import pandas as pd
import random
from pathlib import Path


def main():
    all_batch_dir = Path(f'annotation/all_batches')
    labeled_df = pd.read_json(all_batch_dir / f'labeled_examples.jsonl', lines=True)
    labeled_df['pairID'] = labeled_df['pairID'].astype(str)

    test_ids = labeled_df.id.sample(5000).tolist()
    test_df = labeled_df.loc[labeled_df['id'].isin(test_ids)]
    train_df = labeled_df.loc[~labeled_df['id'].isin(test_ids)]
    
    train_df.to_json(all_batch_dir / f'train.jsonl', lines=True, orient='records')
    test_df.to_json(all_batch_dir / f'test.jsonl', lines=True, orient='records')


if __name__ == "__main__":
    main()