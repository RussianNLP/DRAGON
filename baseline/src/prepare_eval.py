import json
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset


def get_mapping(mapping_texts_ds):
    text_mapping = dict()
    for item in mapping_texts_ds:
        text_mapping[item['id']] = item['public_id']
    return text_mapping


def collect_text_ids(text_ids, mapping=lambda x: x):
    text_ids = json.loads(text_ids)
    if isinstance(text_ids[0], list):
        a = set()
        for item in text_ids:
            a.update(item)
    else:
        a = set(text_ids)
    return [mapping[item] for item in a]


def gen_context(match_ids, pub_texts):
    return '\n\n'.join(pub_texts.loc[match_ids, 'text'].tolist())


def get_private_data(gen_path: Path,
                     output_path: Path,
                     version: str,
                     cache_dir: Path,
                     pub_texts: str,
                     priv_texts: str,
                     priv_qa: str):
    priv_texts = load_dataset(
        priv_texts,
        revision=version,
        cache_dir=cache_dir,
        split='train'
    )
    priv_qa = load_dataset(
        priv_qa,
        revision=version,
        cache_dir=cache_dir,
        split='train'
    ).to_pandas()
    pub_texts = load_dataset(
        pub_texts,
        revision=version,
        cache_dir=cache_dir,
        split='train'
    ).to_pandas()

    mapping = get_mapping(priv_texts)
    priv_qa['text_ids'] = priv_qa['text_ids'].apply(
        collect_text_ids, mapping=mapping
    )

    cols = ['public_id',
            'text_ids',
            'answer',
            'type',
            'text']
    gen_df = pd.read_json(gen_path, orient='records', lines=True)
    fdf = pd.merge(
        gen_df, 
        priv_qa[cols],
        left_on='qid',
        right_on='public_id',
        how='left'
    )
    fdf['context'] = fdf['match_ids'].apply(
        gen_context, pub_texts=pub_texts
    )
    fdf.to_json(
        output_path,
        orient='records',
        lines=True,
        force_ascii=False
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--cache_dir', type=Path, default='./cache')
    parser.add_argument(
        '--pub_texts',
        type=str,
        default='ai-forever/test-rag-bench-public-texts'
    )
    parser.add_argument(
        '--priv_texts',
        type=str,
        default='ai-forever/test-rag-bench-private-texts'
    )
    parser.add_argument(
        '--priv_qa',
        type=str,
        default='ai-forever/test-rag-bench-private-qa'
    )
    args = parser.parse_args()

    gen_path = args.output_dir / args.version / 'gen_output.json'
    output_path = args.output_dir / args.version / 'private_data.json'

    get_private_data(
        gen_path,
        output_path,
        args.version,
        args.cache_dir,
        args.pub_texts,
        args.priv_texts,
        args.priv_qa
    )

