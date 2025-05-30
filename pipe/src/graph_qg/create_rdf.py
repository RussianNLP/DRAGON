import pandas as pd

from pathlib import Path
from rdflib import Graph, Namespace
from argparse import ArgumentParser

from src.lm.utils import parse_json


def split_list_relations(df):
    df = df.reset_index(names='art_id')
    lst_mask = (df.map(lambda x: type(x).__name__) == 'list').any(axis=1)
    lst_df = df.loc[lst_mask].copy()
    df = df.drop(lst_df.index)
    li = df.index[-1]
    new_rels = pd.DataFrame([], columns=df.columns)
    for i, aid, ss, r, os in lst_df.itertuples():
        if not isinstance(ss, list):
            ss = [ss]
        if not isinstance(os, list):
            os = [os]
        for s in ss:
            for o in os:
                new_rels.loc[li] = [aid, s, r, o]
                li += 1

    df = pd.concat([df, new_rels])
    df = df.set_index('art_id').sort_index()
    df.index.name = None
    return df


def collect_entities(df):
    relations = df['relation'].reset_index().rename(
        columns={'relation': 'ent', 'index': 'article_id'}
    )
    relations['type'] = 'rel'
    subjs = df['subject'].reset_index().rename(
        columns={'subject': 'ent', 'index': 'article_id'}
    )
    subjs['type'] = 'ent'
    objs = df['object'].reset_index().rename(
        columns={'object': 'ent', 'index': 'article_id'}
    )
    objs['type'] = 'ent'
    ents = pd.concat([subjs, objs, relations])
    ents = ents.groupby('ent').agg(
        {'article_id': set, 'type': 'first'}
    ).reset_index().reset_index()
    ents['article_id'] = ents['article_id'].map(list)
    ents.set_index('ent', inplace=True)
    return ents


def create_mappings(df, ents):
    df = df.join(
        ents['index'],
        how='left',
        on='subject',
        validate='many_to_one'
    ).rename(columns={'index': 's_idx'})
    df['s_idx'] = df['s_idx'].map(str)

    df = df.join(
        ents['index'],
        how='left',
        on='relation',
        validate='many_to_one'
    ).rename(columns={'index': 'r_idx'})
    df['r_idx'] = df['r_idx'].map(str)

    df = df.join(
        ents['index'],
        how='left',
        on='object',
        validate='many_to_one'
    ).rename(columns={'index': 'o_idx'})
    df['o_idx'] = df['o_idx'].map(str)

    return df


def create_graph(df):
    ents = Namespace('news://ent#')
    refs = Namespace('news://ref#')
    comp = Namespace('news://comp#')

    g = Graph()
    g.bind('ent', ents)
    g.bind('ref', refs)
    g.bind('comp', comp)
    for i, row in df.iterrows():
        if isinstance(row['subject'], (int, float)):
            o = comp[row['s_idx']]
        else:
            s = ents[row['s_idx']]
        r = refs[row['r_idx']]
        if isinstance(row['object'], (int, float)):
            o = comp[row['o_idx']]
        else:
            o = ents[row['o_idx']]
        g.add((s, r, o))
    return g


def run(raw_path, out_dir, colname):
    ents_path = out_dir / 'ents.csv'
    graph_path = out_dir / 'graph.ttl'

    if raw_path.suffix == '.csv':
        raw_df = pd.read_csv(raw_path)
    else:
        raw_df = pd.read_json(raw_path)
    raw_df = raw_df.reset_index()

    raw_df['graph'] = raw_df.apply(
        parse_json,
        axis=1,
        out_cols=('subject', 'relation', 'object'),
        colname=colname
    )
    rels = raw_df['graph'].explode().dropna()
    df = pd.DataFrame(rels.tolist(), index=rels.index)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(df)
    df = df.dropna()
    df.to_csv(out_dir / 'relations.csv')

    ents = collect_entities(df)
    index_df = create_mappings(df, ents)
    graph = create_graph(index_df)
    ents.to_csv(ents_path)
    graph.serialize(graph_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--raw_path', required=True, type=Path)
    parser.add_argument('--out_dir', required=True, type=Path)
    parser.add_argument('--colname', required=True, type=str)
    args = parser.parse_args()

    run(args.raw_path, args.out_dir, args.colname)
