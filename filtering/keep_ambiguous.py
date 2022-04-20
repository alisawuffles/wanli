"""
filter to keep examples with the highest estimated max variability
"""

from utils.constants import NLI_LABELS
import pandas as pd
import numpy as np
import click
import os
from pathlib import Path
import random


@click.command()
@click.option('--data_file', type=str, help='jsonl file')
@click.option('--td_metrics_file', type=str, help='jsonl file')
@click.option('--q', type=float, help='fraction of examples to keep in the filtering stage', default=0.5)
def main(data_file: str, td_metrics_file: str, q: float):
    dataset_df = pd.read_json(data_file, lines=True)
    output_dir = Path(os.path.dirname(data_file))
    td_metrics = pd.read_json(td_metrics_file, lines=True)

    dataset_size = int(len(dataset_df) * q)         # number of examples to keep
    examples_per_label = dataset_size // 3          # number of examples to keep of each label

    # keep only the ambiguous examples
    sorted_ambiguity = td_metrics.sort_values(by='max_variability', ascending=False, axis=0).index
    idxs = []
    for label in NLI_LABELS:
        keep_idxs = dataset_df.loc[sorted_ambiguity].loc[dataset_df['label'] == label][:examples_per_label].index.tolist()
        idxs.extend(keep_idxs)
    
    random.shuffle(idxs)
    dataset_df.loc[idxs].to_json(output_dir / 'ambiguous_examples.jsonl', orient='records', lines=True)
    print(f'Ambiguous data written to {output_dir}/ambiguous_examples.jsonl')


if __name__ == "__main__":
    main()