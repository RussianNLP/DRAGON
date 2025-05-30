import json
import logging
import pandas as pd

from os import path
from datasets import load_dataset
from rdflib import Graph

from src.core.QAGenerator import Generator
from src.utils import Entity
from src.lm.lm_extraction import extract
from src.lm.utils import parse_json, diff
from src.lm.retrieve import build_entity_mapping
from src.graph_qg.create_rdf import (
    collect_entities,
    create_graph,
    create_mappings,
    split_list_relations
)
from src.graph_qg import sample
from src.process.post_process import join_with_texts

logger = logging.getLogger(__name__)


def _generate_syntax(item, ids, columns, qtypes):
    new = {
        'idx': [],
        'sent_idx': [],
        'link': [],
        'text': [],
        'type': [],
        'len': [],
        'qw': [],
        'aw': [],
        'question': [],
        'answer': []
    }
    sidx = 0
    for column in columns:
        extract_col = ['link', column]
        if f'ids_{column}' in item:
            extract_col.append(f'ids_{column}')
        for link, text, *other, idx in zip(*map(
                    lambda x: item[x],
                    extract_col
                ), ids):
            for gen in Generator.create_from_doc(text, qtypes):
                if other:
                    clusters = Entity.parse_entity_clusters(text, other[-1])
                    gen.sow_entities(clusters)
                tqas = gen.generate()
                for tqa in tqas:
                    t, q, a = tqa
                    new['idx'].append(idx)
                    new['sent_idx'].append(sidx)
                    new['link'].append(link)
                    new['text'].append(text)
                    new['type'].append(t)
                    new['len'].append(len(q.split(' ')))
                    new['qw'].append(tqa.qw)
                    new['aw'].append(tqa.aw)
                    new['question'].append(q)
                    new['answer'].append(a)
                sidx += 1
    return new


def generate_syntax(config):
    logger.info(f'Number of files: {len(config.data_files)}')
    dts = load_dataset(
        config.format,
        data_files=config.data_files,
        split='train'
    )
    logger.info(f'Number of rows: {len(dts)}')

    dts = dts.map(
        _generate_syntax,
        batched=True,
        fn_kwargs={
            'columns': config.columns,
            'qtypes': config.syntax_config.qtypes
        },
        remove_columns=dts.column_names,
        num_proc=10,
        with_indices=True
    )
    config.save(dts, name='syntax')
    return dts


def extract_graph(graph_config, model=None, tokenizer=None):
    graphs, entities = {}, {}
    if graph_config.rebuild:
        logger.info('Extract graph')

        logger.info('Graph Stage 1')
        extract(graph_config.lm_config, model=model, tokenizer=tokenizer)
        logger.info('Graph Stage 2')
        build_entity_mapping(
            graph_config.st1_path,
            graph_config.st2_path,
            relation_mapping=graph_config.relation_mapping,
            entity_mapping=graph_config.entity_mapping,
            relation_index=graph_config.relation_index,
            entity_index=graph_config.entity_index
        )
        logger.info('Graph Stage 3')
        extract(graph_config.lm_config2, model=model, tokenizer=tokenizer)

        raw_df = pd.read_json(graph_config.st1_path)
        raw_df = raw_df.reset_index()

        raw_df['raw_graph'] = raw_df.apply(
            parse_json,
            axis=1,
            out_cols=('subject', 'relation', 'object')
        )

        wd_df = pd.read_json(graph_config.st3_path)
        wd_df = wd_df.reset_index()

        wd_df['wikidata_graph'] = wd_df.apply(
            parse_json,
            axis=1,
            out_cols=('subject', 'relation', 'object')
        )

        raw_df['wikidata_graph'] = wd_df['wikidata_graph']
        raw_df['filtered_graph'] = raw_df.apply(
            diff,
            axis=1
        )

        for graph_type in ['wikidata_graph',
                           'filtered_graph']:
            rels = raw_df[graph_type].explode().dropna()
            types = rels.map(lambda x: type(x).__name__)
            rels = rels.loc[types == 'dict']
            df = pd.DataFrame(rels.tolist(), index=rels.index)
            df = df.apply(pd.to_numeric, errors='coerce').fillna(df)
            df = df.map(
                lambda x: int(x)
                if isinstance(x, float) and x.is_integer()
                else x
            )
            df = df.dropna()
            df = split_list_relations(df)
            mask = df['subject'] == df['object']
            df = df.loc[~mask]
            df.to_csv(path.join(graph_config.graph_dir,
                                graph_type + '_relations.csv'))

            ents = collect_entities(df)
            index_df = create_mappings(df, ents)
            graph = create_graph(index_df)

            if graph_type == 'wikidata_graph':
                ents.to_csv(graph_config.wikidata_ents_path)
                graph.serialize(graph_config.wikidata_graph_path)
            else:
                ents.to_csv(graph_config.filtered_ents_path)
                graph.serialize(graph_config.filtered_graph_path)

            ents = ents.reset_index().set_index('index')
            ents['article_id'] = ents['article_id'].map(
                lambda x: set(x)
            )
            graphs[graph_type] = graph
            entities[graph_type] = ents

    else:
        logger.info('Load graph')
        graph = Graph()
        graph.parse(graph_config.filtered_graph_path)
        ents = pd.read_csv(graph_config.filtered_ents_path)
        ents = ents.set_index('index')
        ents['article_id'] = ents['article_id'].map(
            lambda x: set(json.loads(x))
        )
        graphs['filtered_graph'] = graph
        entities['filtered_graph'] = ents

    return graphs['filtered_graph'], entities['filtered_graph']


def generate_from_graph(config,
                        graph,
                        ents,
                        model=None,
                        tokenizer=None,
                        news_df=None):
    logger.info(f'Generate {config.type} questions')
    if config.rebuild_task or not config.task_path.is_file():
        logger.info('Generate task file')
        try:
            query = getattr(sample, f'{config.type}_query')
            res = query(graph, ents)
            res = sample.filter_questions(res, config.n)
            if news_df is not None:
                res = join_with_texts(res, news_df)

            res.to_csv(config.task_path, index=False)
        except ValueError:
            logger.info(f'Cannot extract {config.type} questions from graph')
            return
    else:
        logger.info('Use existing task file')

    extract(config.lm_config, model=model, tokenizer=tokenizer)

    raw_df = pd.read_json(config.raw_path)

    raw_df = raw_df.reset_index()
    raw_df['qas'] = raw_df.apply(
        parse_json,
        axis=1,
        out_cols=('question', 'answer')
    )

    qas = raw_df['qas']
    mask_list = qas.map(lambda x: isinstance(x, list))
    list_series = qas.loc[mask_list].explode()
    qas = pd.concat(
        [qas.loc[~mask_list], list_series], sort=False
    ).sort_index().dropna()

    df = pd.DataFrame(qas.tolist(), index=qas.index)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(df)
    df = df.dropna()
    columns = [
        k for k in config.lm_config.prompt_subs.keys()
        if k in raw_df.columns
    ]
    df = raw_df[columns].join(df, how='inner')
    logger.info(f'Generated {len(df)} {config.type} questions.')
    df.to_csv(config.result_path, index=False)
    # df = pd.read_csv(config.result_path)
    return df
