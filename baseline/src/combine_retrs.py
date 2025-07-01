import json
import numpy as np
import pandas as pd
import itertools

from pathlib import Path
from datasets import load_dataset
from argparse import ArgumentParser


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
         cache_dir,
         pub_texts,
         pub_questions):

    pub_texts_df = load_dataset(pub_texts, revision=version, cache_dir=cache_dir, split='train').to_pandas()
    pub_questions_df = load_dataset(pub_questions, revision=version, cache_dir=cache_dir, split='train').to_pandas()

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

    df['question'] = pub_questions_df.loc[df.index, 'question'].values
    df['context'] = df['match_ids'].apply(lambda x: pub_texts_df.loc[pub_texts_df['id'].isin(x), 'text'].values)
    df['context'] = df['context'].apply(lambda x: '\n\n'.join(x))
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--version', type=str, default='1.11.0')
    parser.add_argument('--cache_dir', type=Path, default='./cache')
    parser.add_argument('--pub_texts', type=str, default="ai-forever/test-rag-bench-public-texts")
    parser.add_argument('--pub_questions', type=str, default="ai-forever/test-rag-bench-public-questions")
    args = parser.parse_args()
    
    retr_dir = args.output_dir / args.version / 'retrievals'
    retr_files = [p for p in retr_dir.glob('*.json')]
    output_file = args.output_dir / args.version / 'gen_input.json'

    main(retr_files, args.version, output_file,
         args.cache_dir, args.pub_texts, args.pub_questions)
