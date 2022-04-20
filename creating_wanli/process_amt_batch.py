"""
process AMT batch results into a jsonl file where each line corresponds to a single annotation
"""

import pandas as pd
from pathlib import Path
from utils.constants import id2label
from utils.utils import example_revised
import ast
import json
import os
from collections import Counter

num_ex_per_hit = 5

def write_feedback(batch_dir, batch_results):
    with open(batch_dir / 'feedback.txt', 'w') as fo:
        for worker_id in set(batch_results.WorkerId):
            worker_df = batch_results.loc[(batch_results['WorkerId'] == worker_id) & (batch_results['Answer.feedback'] != '{}')]
            if len(worker_df.index) == 0:
                continue
            fo.write('-----------------\n')
            fo.write(f'Worker ID: {worker_id}\n')
            for i, row in worker_df.iterrows():
                if isinstance(row['Answer.feedback'], float):
                    continue
                # print the relevant question
                for j in range(1,6):
                    if str(j) in row["Answer.feedback"] or (j == 1 and 'first' in row["Answer.feedback"]) or (j == 1 and 'number one' in row["Answer.feedback"]) or (j == 5 and 'last' in row["Answer.feedback"]):
                        fo.write(f'{j}) Premise: {row[f"Input.premise{j}"]}\n')
                        fo.write(f'Hypothesis: {row[f"Input.hypothesis{j}"]}\n')
                        if row[f"Input.premise{j}"] != row[f"Answer.premise{j}"]:
                            fo.write(f'Revised premise: {row[f"Answer.premise{j}"]}\n')
                        if row[f"Input.hypothesis{j}"] != row[f"Answer.hypothesis{j}"]:
                            fo.write(f'Revised hypothesis: {row[f"Answer.hypothesis{j}"]}\n')
                        fo.write(f'Relationship: {id2label[row[f"Answer.q{j}_gold"]]}\n')
                fo.write(f'Feedback: {row["Answer.feedback"]}\n\n')


def process_batch(subdir: str):
    batch_id = subdir.split('_')[1]
    batch_dir = Path(f'annotation') / subdir
    batch_results = pd.read_csv(batch_dir / f'Batch_{batch_id}_batch_results.csv')

    # summarize feedback in output file
    write_feedback(batch_dir, batch_results)

    # create a json file where each row corresponds to a label example
    qualified_workers = pd.read_csv(batch_dir / 'qualified_workers.csv')['worker_id'].to_list()
    labeled_ex = []
    # for each HIT
    for i, row in batch_results.iterrows():
        if row['AssignmentStatus'] == 'Rejected':
            continue
        if row['WorkerId'] not in qualified_workers:
            continue
        hit_info = {
            'WorkerId': row['WorkerId'],
            'HITId': row['HITId'],
            'TimeOnPageInSeconds': row['Answer.ee']
        }
        # for each example in the HIT
        for j in range(1, num_ex_per_hit+1):
            ex = hit_info.copy()
            ex['id'] = row[f'Input.id{j}']
            ex['nearest_neighbors'] = ast.literal_eval(row[f'Input.nearest_neighbors{j}'])
            ex['premise'] = row[f'Input.premise{j}']
            ex['hypothesis'] = row[f'Input.hypothesis{j}']
            ex['label'] = row[f'Input.label{j}']
            ex['revised_premise'] = row[f'Answer.premise{j}']
            ex['revised_hypothesis'] = row[f'Answer.hypothesis{j}']
            ex['gold'] = id2label[row[f'Answer.q{j}_gold']]
            ex['revised'] = example_revised(ex)
            labeled_ex.append(ex)

    results_df = pd.DataFrame(labeled_ex)
    print(f'Processed {len(results_df)} annotations from {batch_dir}')
    results_df.to_json(batch_dir / f'processed_results.jsonl', lines=True, orient='records')


def main():
    # process results from each batch and combine results across all batches
    dfs = []
    for subdir in os.listdir(f'annotation/'):
        if subdir.startswith('batch_'):
            process_batch(subdir)
            dfs.append(pd.read_json(f'annotation/{subdir}/processed_results.jsonl', lines=True))
    concatenated = pd.concat(dfs).sample(frac=1)

    # write output
    concatenated.to_json(f'annotation/all_batches/processed_results.jsonl', orient='records', lines=True)
    with open(f'annotation/all_batches/processed_results.json', 'w') as fo:
        json.dump(concatenated.to_dict(orient='records'), fo, indent=4)


if __name__ == "__main__":
    main()