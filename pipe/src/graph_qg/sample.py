import numpy as np
import pandas as pd

from rdflib import Graph

from src.graph_qg.sparql_queries import (
    SIMPLE_QUERY,
    SET_QUERY,
    COND_QUERY,
    MH_QUERY,
    COMP_QUERY
)


def get_id(x):
    if isinstance(x, str):
        if '|' in x:
            x = [
                int(y.split('#')[-1])
                for y in x.split('|')
            ]
        else:
            x = int(x.split('#')[-1])
    return x


def id_to_name(df, entdf, columns):
    for c in columns:
        df[c] = df[c].map(entdf['ent'])
    return df


def get_articles(df, entdf, columns):
    aidcols = []
    for c in columns:
        aidcol = f'{c}_aid'
        df[aidcol] = df[c].map(entdf['article_id'])
        aidcols.append(aidcol)
    return df


def _query(g: Graph, entdf: pd.DataFrame, q: str):
    qres = g.query(q)
    res = pd.DataFrame(map(lambda x: x.asdict(), qres))
    res = res.map(get_id)
    mask = res.apply(
        lambda xs: frozenset([x for x in xs if x == x]), 1
    ).duplicated()
    res = res.loc[~mask]
    cols = res.columns
    res = get_articles(res, entdf, cols)
    res = id_to_name(res, entdf, cols)
    return res


def filter_questions(df, n=50):
    if n == -1:
        return df
    ids = np.random.choice(df.index, min(n, len(df)), replace=False)
    df = df.loc[ids]
    return df


def simple_query(g: Graph, entdf: pd.DataFrame):
    res = _query(g, entdf, SIMPLE_QUERY)

    def trans(row):
        s = [(row['s'], row['r'], row['o'])]
        s = map(lambda x: '|'.join(map(str, x)), s)
        return '\n'.join(map(lambda x: f'({x})', s))

    def get_ein(row):
        es = list(set((row['s'], row['o'])))
        return es

    res['sample'] = res.apply(trans, axis=1)
    res['aid'] = res[['s_aid', 'r_aid', 'o_aid']].apply(
        lambda x: list(set.intersection(*x.values)),
        axis=1
    )
    res['ein'] = res.apply(get_ein, axis=1)
    return res


def set_query(g: Graph, entdf: pd.DataFrame):
    qres = g.query(SET_QUERY)
    res = pd.DataFrame(map(lambda x: x.asdict(), qres))
    if not len(res):
        raise ValueError('Cannot extract relations for SET question')
    res = res.map(get_id)
    res = res.explode('s')
    res = res.explode('o')
    cols = ['s', 'r', 'o']
    res = get_articles(res, entdf, cols)
    res = id_to_name(res, entdf, cols)

    def trans(subdf):
        strs = []
        aids = None
        subj, rel, obj = set(), set(), set()
        for i, row in subdf.iterrows():
            ln = row['len']
            s = row['s']
            r = row['r']
            o = row['o']
            subj.add(s)
            rel.add(r)
            obj.add(o)
            _aids = set(row['s_aid'])
            _aids &= set(row['r_aid'])
            _aids &= set(row['o_aid'])
            if aids is None:
                aids = _aids
            else:
                aids |= _aids
            strs.append(f'({s}|{r}|{o})')

        subjects = list(subj)
        objects = list(obj)

        return pd.DataFrame({
            # 's': [list(subj) if len(subj) > 1 else subj.pop()],
            's': [subjects if len(subjects) > 1 else subjects[0]],
            'r': [rel.pop()],
            # 'o': [list(obj) if len(obj) > 1 else obj.pop()],
            'o': [objects if len(objects) > 1 else objects[0]],
            'eina': [objects if len(objects) > 1 else subjects],
            'einq': [subjects if len(objects) > 1 else objects],
            'len': [ln],
            'sample': ['\n'.join(strs)],
            'aid': [list(aids)]
        })

    res = res.groupby(level=-1).apply(trans)
    return res


def cond_query(g: Graph, entdf: pd.DataFrame):
    res = _query(g, entdf, COND_QUERY)

    def trans(row):
        if 'b' in row and row['b'] is not np.nan:
            s = [
                (row['b'], row['r'], row['o']),
                (row['s'], row['r1'], row['b']),
            ]
        elif 's1' in row and row['s1'] is not np.nan:
            s = [
                (row['s'], row['r'], row['o']),
                (row['s1'], row['r1'], row['o']),
            ]
        else:
            s = [
                (row['s'], row['r'], row['o']),
                (row['s'], row['r1'], row['o1']),
            ]
        s = map(lambda x: '|'.join(map(str, x)), s)
        return '\n'.join(map(lambda x: f'({x})', s))

    def get_aid(row):
        if 'b' in row and row['b'] is not np.nan:
            _aids = [
                (row['b_aid'], row['r_aid'], row['o_aid']),
                (row['s_aid'], row['r1_aid'], row['b_aid']),
            ]
        elif 's1' in row and row['s1'] is not np.nan:
            _aids = [
                (row['s_aid'], row['r_aid'], row['o_aid']),
                (row['s1_aid'], row['r1_aid'], row['o_aid']),
            ]
        else:
            _aids = [
                (row['s_aid'], row['r_aid'], row['o_aid']),
                (row['s_aid'], row['r1_aid'], row['o1_aid']),
            ]
        aids = list(map(
            lambda x: list(set.intersection(*x)),
            _aids
        ))
        return aids

    def get_ein(row):
        if 'b' in row and row['b'] is not np.nan:
            sent = [row['b']]
            es = [row['o'], row['s']]
        elif 's1' in row and row['s1'] is not np.nan:
            sent = [row['o']]
            es = [row['s'], row['s1']]
        else:
            sent = [row['s']]
            es = [row['o'], row['o1']]
        return sent, es

    res['sample'] = res.apply(trans, axis=1)
    res['aid'] = res.apply(get_aid, axis=1)
    res['eina'], res['einq'] = zip(
        *res.apply(get_ein, axis=1)
    )
    return res


def mh_query(g: Graph, entdf: pd.DataFrame):
    res = _query(g, entdf, MH_QUERY)

    def trans(row):
        if 'b' in row and row['b'] is not np.nan:
            s = [
                (row['b'], row['r'], row['o']),
                (row['s'], row['r1'], row['b']),
            ]
        elif 's1' in row and row['s1'] is not np.nan:
            s = [
                (row['s'], row['r'], row['o']),
                (row['s1'], row['r1'], row['o']),
            ]
        else:
            s = [
                (row['s'], row['r'], row['o']),
                (row['s'], row['r1'], row['o1']),
            ]
        s = map(lambda x: '|'.join(map(str, x)), s)
        return '\n'.join(map(lambda x: f'({x})', s))

    def get_aid(row):
        if 'b' in row and row['b'] is not np.nan:
            _aids = [
                (row['b_aid'], row['r_aid'], row['o_aid']),
                (row['s_aid'], row['r1_aid'], row['b_aid']),
            ]
        elif 's1' in row and row['s1'] is not np.nan:
            _aids = [
                (row['s_aid'], row['r_aid'], row['o_aid']),
                (row['s1_aid'], row['r1_aid'], row['o_aid']),
            ]
        else:
            _aids = [
                (row['s_aid'], row['r_aid'], row['o_aid']),
                (row['s_aid'], row['r1_aid'], row['o1_aid']),
            ]
        aids = list(map(
            lambda x: list(set.intersection(*x)),
            _aids
        ))
        return aids

    def get_ein(row):
        if 'b' in row and row['b'] is not np.nan:
            sent = [row['b']]
            es = [row['o'], row['s']]
        elif 's1' in row and row['s1'] is not np.nan:
            sent = [row['o']]
            es = [row['s'], row['s1']]
        else:
            sent = [row['s']]
            es = [row['o'], row['o1']]
        return sent, es

    res['sample'] = res.apply(trans, axis=1)
    res['aid'] = res.apply(get_aid, axis=1)
    res['sent'], res['ein'] = zip(*res.apply(get_ein, axis=1))
    return res


def comp_query(g: Graph, entdf: pd.DataFrame):
    res = _query(g, entdf, COMP_QUERY)

    def trans(row):
        s = [
            (row['s'], row['r'], row['o']),
            (row['s1'], row['r'], row['o1'])
        ]
        s = map(lambda x: '|'.join(map(str, x)), s)
        return '\n'.join(map(lambda x: f'({x})', s))

    def get_aid(row):
        _aids = [
            (row['s_aid'], row['r_aid'], row['o_aid']),
            (row['s1_aid'], row['r_aid'], row['o1_aid'])
        ]
        aids = list(map(
            lambda x: list(set.intersection(*x)),
            _aids
        ))
        return aids

    def get_ein(row):
        return [row['s'], row['s1']]

    res['sample'] = res.apply(trans, axis=1)
    res['aid'] = res.apply(get_aid, axis=1)
    res['ein'] = res.apply(get_ein, axis=1)
    return res
