import logging
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

from src.process.config import Config

logger = logging.getLogger(__name__)


def prepare_simple(df, config, filters):
    models = [c.column_name for c in config]
    metrics = [f.type for f in filters]
    columns = [f'{met}-{mod}' for met in metrics for mod in models]
    df['is_simple'] = df[columns].max(axis=1)
    return df


def prepare_judge(df, judge_path):
    logger.info('judge')
    jdf = pd.read_csv(judge_path)
    _jdf = jdf.loc[:, list(
        filter(lambda x: x.endswith('_score'), jdf.columns)
    )]
    df = pd.concat((df, _jdf), axis=1)
    return df


def filt(df, cfg):
    ops = {
        'gt': lambda x, t: x > t,
        'lt': lambda x, t: x < t,
        'ge': lambda x, t: x >= t,
        'le': lambda x, t: x <= t,
    }
    how = cfg.get('how', 'lt')
    mask = df[cfg['columns']].map(
        lambda x: ops[how](x, cfg['thr'])
    ).all(axis=1)
    return df.loc[mask]


def filter_peaks(df, cols, quant=None):
    for col in cols:
        if quant is None:
            mx = df[col].max()
            mn = df[col].min()
        else:
            mx = df[col].quantile(1-quant)
            mn = df[col].quantile(quant)
        df = df.loc[(df[col] > mn) & (df[col] < mx)]
    return df


def sort_f1(df, cols, ascending):
    sortdf = pd.DataFrame()
    if not isinstance(ascending, list):
        ascending = [ascending] * len(cols)
    for c, a in zip(cols, ascending):
        if a:
            sortdf[c] = (df[c].max() - df[c]) / (df[c].max() - df[c].min())
        else:
            sortdf[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

    sortdf['hmean'] = sortdf.apply(
        lambda x: x.shape[0] / np.sum(1 / x), axis=1
    )
    sortdf = sortdf.sort_values('hmean', ascending=False)
    df = df.loc[sortdf.index]
    return df


def sort_dev(df, cols, ascending):
    sortcols = []
    for c in cols:
        mean = df[c].mean()
        sortcol = f'{c}_dev'
        sortcols.append(sortcol)
        df[sortcol] = (df[c] - mean).abs()
    return df.sort_values(sortcols, ascending=ascending)


def sort_dev_f1(df, cols, ascending=True):
    sortdf = pd.DataFrame()
    for c in cols:
        mean = df[c].mean()
        sortdf[c] = (df[c] - mean).abs()
        sortdf[c] = (
            (sortdf[c] - sortdf[c].min()) /
            (sortdf[c].max() - sortdf[c].min())
        )

    sortdf['hmean'] = sortdf.apply(
        lambda x: x.shape[0] / np.sum(1 / x), axis=1
    )
    sortdf = sortdf.sort_values('hmean', ascending=ascending)
    df = df.loc[sortdf.index]
    return df


_sort_funcs = {
    'f1': sort_f1,
    'dev': sort_dev,
    'dev_f1': sort_dev_f1
}


def category_filter(df, flts):
    logger.info('len before:', len(df))
    for flt in flts:
        if len(df) == 0:
            break
        if 'columns' in flt:
            columns = flt['columns']
            if not isinstance(columns, list):
                columns = [columns]
                flt['columns'] = columns
        if flt['type'] == 'filter':
            df = filt(df, flt)
        elif flt['type'] == 'filter_peaks':
            df = filter_peaks(df, flt['columns'], flt['q'])
        elif flt['type'] == 'select':
            mask = (df[flt['columns']] == flt['val']).all(axis=1)
            df = df.loc[mask]
        elif flt['type'] == 'sort':
            ascend = flt.get('ascending', False)
            func = flt.get('func', None)
            if func is None:
                df = df.sort_values(flt['columns'], ascending=ascend)
            else:
                func = _sort_funcs[func]
                df = func(df, flt['columns'], ascend)
        elif flt['type'] == 'topn':
            df = df.iloc[:flt['n']]
        logger.info(len(df))
    return df


def main(config, data_path, output_path):
    df = pd.read_csv(data_path)

    if config.answer_generations:
        df = prepare_simple(df,
                            config.answer_generations,
                            config.ag_filters)
    if config.judge_instructions:
        judge_path = config.output_dir / 'judges.csv'
        df = prepare_judge(df, judge_path)
    logger.info('Number of questions:', len(df))

    dfs = []
    for cat, flt in config.filter_config.items():
        logger.info('Filter category:', cat)
        subdf = df.loc[df['type'] == cat]
        subdf = category_filter(subdf, flt)
        dfs.append(subdf)

    df = pd.concat(dfs)
    logger.info('Final len:', len(df))
    df.to_csv(output_path, index=False)


def run():
    parser = ArgumentParser()
    parser.add_argument('global_config', type=Path)
    args = parser.parse_args()

    config = Config.parse(args.global_config)
    data_path = config.output_dir / 'processed.csv'
    output_path = config.output_dir / 'filtered.csv'

    if config.filter_config:
        main(config, data_path, output_path)


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        logger.error(e, exc_info=True)
        raise
