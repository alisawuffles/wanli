import os
import json
import click
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import spatial
from generation.context_formats import format_incontext_examples, label_to_instruction
from generation.gpt3_generation import request
from utils.utils import ensure_dir
from utils.constants import NLI_LABELS


@click.command()
@click.option('--model_path', type=str, help='contains pre-computed representations and training dynamics')
@click.option('--num_gens_per_prompt', default=5, type=int, help='')
@click.option('--num_incontext_examples', default=5, type=int)
@click.option('--ambiguity_quantile', default=0.75, type=float,
    help='set to 0 for random seed examples'
)
@click.option('--num_examples', default=10, type=int, 
    help='total number of generated examples desired, including previously generated examples'
)
def main(
    model_path: str,
    num_gens_per_prompt: int, 
    num_incontext_examples: int,
    ambiguity_quantile: float,
    num_examples: int,
):
    output_dir = Path(f'generated_data/n{num_gens_per_prompt}_nn')
    model_path = Path(model_path)
    ensure_dir(output_dir)

    # load previous generations if they exist
    if os.path.exists(output_dir / 'examples.jsonl'):
        previous_gens = pd.read_json(output_dir / 'examples.jsonl', lines=True)
        indices_to_skip = set([ns[0] for ns in previous_gens['nearest_neighbors']])
        generated_examples = previous_gens.to_dict('records')
        print(f'Generations file already contains {len(indices_to_skip)} examples')
    else:
        generated_examples = []
        indices_to_skip = []

    # pre-computed embeddings of training examples
    with open(model_path / 'representations/mnli.npy', 'rb') as fin:
        mnli_vectors = np.load(fin)
        tree = spatial.KDTree(mnli_vectors)
    
    # load pool of MNLI data
    mnli = pd.read_json('data/mnli/train.jsonl', lines=True, orient='records')
    td_metrics = pd.read_json(model_path / 'training_dynamics/td_metrics.jsonl', lines=True)
    mnli['variability'] = td_metrics['variability'].tolist()
    # skip telephone genre
    mnli = mnli.loc[mnli['genre'] != 'telephone']
    # get the most ambiguous examples within each label class
    ambiguous_dfs = []
    for label in NLI_LABELS:
        label_df = mnli.loc[mnli['gold'] == label]
        thres = label_df['variability'].quantile(q=ambiguity_quantile)
        ambiguous_dfs.append(label_df[label_df['variability'] > thres])
    ambiguous_mnli = pd.concat(ambiguous_dfs)
    # shuffle and skip examples we've used before
    ambiguous_mnli = ambiguous_mnli.sample(frac=1)
    ambiguous_mnli = ambiguous_mnli.drop(indices_to_skip)
    # write output continuously and flush periodically
    examples_fo = open(output_dir / 'examples.jsonl', 'w')
    lines_per_flush = 100

    # generate examples!
    pbar = tqdm(initial=len(indices_to_skip), total=num_examples, position=0, leave=True)
    for _, row in ambiguous_mnli.iterrows():
        id = row['id']
        label = mnli.loc[id]['gold']
        
        # get nearest neighbors
        embedding = mnli_vectors[id,:]
        neighbor_ids = tree.query(embedding, k=15)[1]
        neighbor_ids = [n for n in neighbor_ids if n in mnli.index]     # some neighbor_ids should be excluded if they are telephone convos
        neighbors_df = mnli.loc[neighbor_ids].loc[mnli['gold'] == label][:num_incontext_examples]
        if len(neighbors_df.index) < num_incontext_examples:
            continue 
        
        context_string = format_incontext_examples(neighbors_df, label=label)
        # write an example context to files
        if not os.path.exists(output_dir / f'{label}_context.txt'):
            with open(output_dir / f'{label}_context.txt', 'w') as template_fo:
                template_fo.write(context_string)
        
        for i in range(num_gens_per_prompt):
            generation = request(
                context_string, 
                engine='curie',
                max_tokens=120,
                top_p=0.5,
                stop='\n\n',
            )
            try:
                premise, hypothesis = generation.split('\n' + label_to_instruction[label]['label'] + ': ')
            except ValueError:
                continue
            generated_ex = {
                'premise': premise,
                'hypothesis': hypothesis,
                'label': label,
                'nearest_neighbors': neighbors_df.index.tolist()
            }
            generated_examples.append(generated_ex)
            pbar.update()
            # write output
            examples_fo.write(json.dumps(generated_ex, default=str) + '\n')
            if len(generated_examples) % lines_per_flush == 0:
                examples_fo.flush()

        if len(generated_examples) >= num_examples:
            break
    
    examples_fo.close()
    
    with open(output_dir / 'examples.json', 'w') as fo:
        json.dump(generated_examples, fo, indent=4)


if __name__ == "__main__":
    main()