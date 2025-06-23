import json
import numpy as np
import pandas as pd
import itertools

from pathlib import Path
from datasets import load_dataset
from argparse import ArgumentParser


_pub_texts = "ai-forever/test-rag-bench-public-texts"
_pub_questions = "ai-forever/test-rag-bench-public-questions"
_priv_texts = "ai-forever/test-rag-bench-private-texts"
_priv_qa = "ai-forever/test-rag-bench-private-qa"


def get_mapping(mapping_texts_ds):
    text_mapping = dict()
    for item in mapping_texts_ds:
        text_mapping[item['id']] = item['public_id']
    return text_mapping


def collect_text_ids(text_ids):
    text_ids = json.loads(text_ids)
    if isinstance(text_ids[0], list):
        a = set()
        for item in text_ids:
            a.update(item)
        return a
    else:
        return set(text_ids)

def main(retr_files,
         version,
         output_file,
         cache_dir):

    pub_texts = load_dataset(_pub_texts, revision=version, cache_dir=cache_dir, split='train').to_pandas()
    pub_questions = load_dataset(_pub_questions, revision=version, cache_dir=cache_dir, split='train').to_pandas()
    priv_texts = load_dataset(_priv_texts, revision=version, cache_dir=cache_dir, split='train').to_pandas()
    priv_qa = load_dataset(_priv_qa, revision=version, cache_dir=cache_dir, split='train').to_pandas()

    priv_qa['text_ids'] = priv_qa['text_ids'].apply(collect_text_ids)

    dfs = []
    for fp in retr_files:
        with open(fp, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(
            itertools.chain.from_iterable(
                map(
                    lambda x: x.items(),
                    data
                )
            ), columns=['qid', 'match_ids']
        )
        df['retriever'] = fp.stem
        dfs.append(df)

    df = pd.concat(dfs)

    df['question'] = pub_questions.loc[df.index, 'question'].values
    df['context'] = df['match_ids'].apply(lambda x: pub_texts.loc[pub_texts['id'].isin(x), 'text'].values)
    df['context'] = df['context'].apply(lambda x: '\n\n'.join(x))
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--version', type=str, default='1.11.0')
    parser.add_argument('--cache_dir', type=Path, default='./cache')
    args = parser.parse_args()
    
    retr_dir = args.output_dir / args.version / 'retrievals'
    retr_files = [p for p in retr_dir.glob('*.json')]
    output_file = args.output_dir / args.version / 'gen_input.json'

    main(retr_files, args.version, output_file, args.cache_dir)
