import os
import random
from re import S
import string

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def flip(p):
    """
    given probability p of returning 1, simulate flip of biased coin
    """
    return 1 if random.random() < p else 0


def strip_punctuation_and_casing(s):
    s = s.lower()   # lower case
    s = s.translate(str.maketrans('', '', string.punctuation))  # strip punctuation
    return s

# return True if the example was neither discarded nor revised
def example_untouched(row):
    if row['gold'] != 'discard' and row['premise'] == row['revised_premise'] and row['hypothesis'] == row['revised_hypothesis']:
        return True
    return False

# return True if the example was not discarded but revised
def example_revised(row):
    if row['gold'] != 'discard' and (row['premise'] != row['revised_premise'] or row['hypothesis'] != row['revised_hypothesis']):
        return True
    return False

def only_punctuation_revised(row):
    if example_revised(row) and strip_punctuation_and_casing(row['premise']).strip() == strip_punctuation_and_casing(row['revised_premise']).strip() and strip_punctuation_and_casing(row['hypothesis']).strip() == strip_punctuation_and_casing(row['revised_hypothesis']).strip():
        return True
    return False

# return 'generated_revised' or 'generated'
def get_genre(row):
    if row['premise'] == row['revised_premise'] and row['hypothesis'] == row['revised_hypothesis']:
        return 'generated'
    return 'generated_revised'