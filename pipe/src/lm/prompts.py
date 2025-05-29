import json

from pathlib import Path


PROMPT_DIR = Path(__file__).parents[2] / 'prompts'


def load_prompt(val):
    prompt = val
    if isinstance(val, str) and val.endswith('.txt'):
        path = PROMPT_DIR / val
        if path.is_file():
            with open(path) as f:
                prompt = f.read()
    elif isinstance(val, str) and val.endswith('.json'):
        path = PROMPT_DIR / val
        if path.is_file():
            with open(path) as f:
                prompt = json.load(f)
    return prompt
