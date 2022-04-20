import pandas as pd

label_to_instruction = {
    'contradiction': {
        'instruction': 'Write a pair of sentences that have the same relationship as the previous examples.',
        'label': 'Contradiction'
    },
    'entailment': {
        'instruction': 'Write a pair of sentences that have the same relationship as the previous examples.',
        'label': 'Implication'
    },
    'neutral': {
        'instruction': 'Write a pair of sentences that have the same relationship as the previous examples.',
        'label': 'Possibility'
    },
}


def format_incontext_examples(
    examples: pd.DataFrame, 
    label: str, 
):
    """
    format a given group of examples as context for GPT-3, using template for the given label
    """
    examples = examples[::-1]
    context_string = f'{label_to_instruction[label]["instruction"]} Examples:\n\n'
    # write in context examples
    for i, (_, row) in enumerate(examples.iterrows()):
        # for every chunk_size examples, repeat instructions and enumerate starting from 1
        context_string += f'{i+1}. {row["premise"]}\n{label_to_instruction[label]["label"]}: {row["hypothesis"]}\n\n'

    # write final numbering and premise, if provided
    context_string += f'{len(examples.index)+1}.'

    return context_string
