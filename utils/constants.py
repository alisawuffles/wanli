from pathlib import Path
import yaml

# Config
CONFIG_FILE = Path('config.yml')
OPENAI_API_KEY = ''
try:
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    OPENAI_API_KEY = config['openai']
except FileNotFoundError:
    print('No config file found. API keys will not be loaded.')

NLI_LABELS = ['contradiction', 'entailment', 'neutral']
id2label = {i: label for i, label in enumerate(NLI_LABELS)}
id2label[3] = 'discard'
SEED = 1