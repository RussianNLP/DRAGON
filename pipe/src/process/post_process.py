import os
import ast
import yaml
import torch
import logging
import pandas as pd

from tqdm import tqdm
from torch.nn.functional import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from argparse import ArgumentParser
from fuzzywuzzy import fuzz
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer

from src.process.config import Config
from src.lm.lm_extraction import extract

logger = logging.getLogger('src.process.post_process')


def join_with_texts(qdf, raw_df):
    new_columns = {
        c: [] for c in raw_df.columns
    }
    index = []

    for i, item in qdf.iterrows():
        aids = item['aid']
        if isinstance(aids, str):
            aids = ast.literal_eval(aids)
        if isinstance(aids[0], list):
            aids = [a for aid in aids for a in aid]
        aids = list(set(aids))
        rows = raw_df.loc[aids]
        index.append(i)
        for k in new_columns:
            new_columns[k].append(
                '\n---\n'.join(rows[k].to_list())
            )

    ndf = pd.DataFrame(new_columns, index=index)
    ndf = pd.concat(
        [ndf, qdf],
        axis=1
    )
    return ndf


def binary_classify(item, text_column, to_column, tokenizer, model, device):
    text = item[text_column]
    inputs = tokenizer(text, return_tensors='pt', padding=True).to(device)
    outputs = model(**inputs)
    prob = softmax(outputs.logits, dim=1).detach().cpu()[:, 1].tolist()
    return {to_column: prob}


def perplexity(item, name, tokenizer, model, device):
    inp = item[name]

    inp_ids = tokenizer(
        inp, return_tensors='pt', padding=True).input_ids.to(device)

    with torch.no_grad():
        out = model(inp_ids, labels=inp_ids).loss.item()

    return out


def pmi(item, qname, aname, tokenizer, model, device):
    q = item[qname]
    a = item[aname]

    a_ids = tokenizer(
        a, return_tensors='pt', padding=True).input_ids.to(device)
    u_ids = tokenizer(
        q + ' ' + a,
        return_tensors='pt',
        padding=True
    ).input_ids.to(device)

    u_labels = u_ids.clone()
    u_labels[:, :-a_ids.size()[-1]] = -100

    with torch.no_grad():
        aout = model(a_ids, labels=a_ids).loss.item()
        cout = model(u_ids, labels=u_labels).loss.item()

    return aout - cout


def get_len(item, cname):
    return len(item[cname])


def get_ein(item, ent_col, find_in):
    ents = item[ent_col]
    if ents is None:
        return -1
    try:
        ents = ast.literal_eval(ents)
    except Exception:
        ents = [ents]
    v = 0
    for ent in ents:
        e = str(ent).lower()
        for cname in find_in:
            s = item[cname].lower()
            v += fuzz.partial_ratio(e, s)
    return v / len(ents)


def match_ner(item, ent_cols, ners):
    c = 0
    for ent_col in ent_cols:
        ents = item[ent_col]
        if ents is None:
            continue
        ents = ast.literal_eval(ents)
        for e in ents:
            if str(e).lower() in ners:
                c += 1
    return c


def get_processor(proc_desc):
    if proc_desc['name'] == 'hf_classify':
        model = AutoModelForSequenceClassification.from_pretrained(
            proc_desc['model_path']
        ).to(proc_desc.get('device', 'cpu'))
        tokenizer = AutoTokenizer.from_pretrained(proc_desc['model_path'])
        return binary_classify, dict(
            text_column=proc_desc['from_column'],
            to_column=proc_desc['to_column'],
            tokenizer=tokenizer,
            model=model,
            device=proc_desc.get('device', 'cpu')
        )
    elif proc_desc['name'] == 'perplexity':
        model = AutoModelForCausalLM.from_pretrained(
            proc_desc['model_path']
        ).to(proc_desc.get('device', 'cpu'))
        tokenizer = AutoTokenizer.from_pretrained(proc_desc['model_path'])
        return perplexity, dict(
            name=proc_desc['from_column'],
            tokenizer=tokenizer,
            model=model,
            device=proc_desc.get('device', 'cpu')
        )
    elif proc_desc['name'] == 'pmi':
        model = AutoModelForCausalLM.from_pretrained(
            proc_desc['model_path']
        ).to(proc_desc.get('device', 'cpu'))
        tokenizer = AutoTokenizer.from_pretrained(proc_desc['model_path'])
        return pmi, dict(
            qname=proc_desc['qname'],
            aname=proc_desc['aname'],
            tokenizer=tokenizer,
            model=model,
            device=proc_desc.get('device', 'cpu')
        )
    elif proc_desc['name'] == 'len':
        return get_len, dict(
            cname=proc_desc['from_column']
        )
    elif proc_desc['name'] == 'ein':
        return get_ein, dict(
            ent_col=proc_desc['ent_col'],
            find_in=proc_desc['find_in'],
        )
    elif proc_desc['name'] == 'match_ner':
        nerdf = pd.read_csv(proc_desc['ner_path'])
        ners = set(nerdf['ner'])

        return match_ner, dict(
            ent_cols=proc_desc['cols'],
            ners=ners
        )
    raise NotImplementedError(
        f'Transformation {proc_desc["name"]} not implemented!'
    )


def or_conditions(conditions):
    def inner(item):
        return any((c(item) for c in conditions))
    return inner


def get_filter(desc):
    conditions = []
    if 'thr' in desc:
        thr = desc['thr']
        col = desc['column_name']
        conditions.append(lambda item: item[col] > thr)
    elif 'val' in desc:
        val = desc['val']
        neq = desc.get('neq', False)
        if not isinstance(val, list):
            val = [val]
        col = desc['column_name']
        conditions.append(
            lambda item: (
                item[col] not in val
                if neq
                else item[col] in val
            )
        )

    return or_conditions(conditions)


def get_selector(dts, desc):
    col = desc['column_name']
    gb = desc['groupby']

    df = dts.select_columns(gb + [col]).to_pandas()
    df = df.reset_index(names='N')

    mask = df['type'] == desc['qtype']
    subdf = df.loc[mask]

    tmp = subdf[gb + [col]].groupby(gb).apply('max').reset_index()
    tmp = tmp.rename(columns={col: 'tmp'})

    tmp = pd.merge(df, tmp, how='left', on=gb)
    res = tmp.loc[(tmp[col] == tmp['tmp']) | ~mask, 'N'].tolist()

    return res


def generate_answers(config, dataset, model=None, tokenizer=None):
    full_df = dataset.to_pandas()
    full_df['answer_words_count'] = full_df['answer'].apply(lambda x: str(len(x.strip().split())))
    for cfg in config:
        gen_result = extract(cfg.lm_config,
                            model=model,
                            tokenizer=tokenizer,
                            dataset=Dataset.from_pandas(full_df, preserve_index=False),
                            write=False)
        answers = [res['model_answer'] for res in gen_result]
        full_df[cfg.column_name] = answers
    fout_pth = os.path.join(cfg.final_res_dir, "with_answers.csv")
    full_df = full_df.drop(['answer_words_count'], axis=1)
    full_df.to_csv(fout_pth, index=False)
    return Dataset.from_pandas(full_df, preserve_index=False)


def filter_questions(df, flt_config):
    for filt in flt_config:
        if filt.type == 'bert_score':
            df = bert_score_filter(df, filt)
    return df


def bert_score_filter(df, flt_config):
    df = df.to_pandas()
    model = SentenceTransformer(flt_config.model_name)
    real_ans_enc = model.encode(df['answer'].values, prompt_name="classification",  convert_to_tensor=True)

    for column_name in flt_config.columns:
        gen_ans_enc = model.encode(df[column_name].values, prompt_name="classification",  convert_to_tensor=True)
        sim_scores = (gen_ans_enc @ real_ans_enc.T).diag().tolist()
        df[f'bert_score-{column_name}'] = sim_scores
        if flt_config.drop:
            df = df[df[f'bert_score-{column_name}'] < flt_config.thr]
            df = df.drop([f'bert_score-{column_name}', column_name], axis=1).reset_index(drop=True)

    return Dataset.from_pandas(df, preserve_index=False)


def post_process(pp_config, dts):
    logger.info('Post process')
    for desc in pp_config:
        if desc['type'] == 'filter':
            flt = get_filter(desc)
            dts = dts.filter(flt)
        if desc['type'] == 'select':
            slt = get_selector(dts, desc)
            dts = dts.select(slt)
        elif desc['type'] == 'transform':
            logger.info(desc['name'])
            func, kwargs = get_processor(desc)
            if 'batch_size' in desc:
                dts = dts.map(
                    func,
                    batched=True,
                    batch_size=desc.get('batch_size', 1),
                    fn_kwargs=kwargs
                )
            else:
                reslist = []
                for item in tqdm(dts, desc=desc['type']):
                    res = func(item, **kwargs)
                    reslist.append(res)
                dts = dts.add_column(desc['to_column'], reslist)
        logger.info(f'Filtered: {len(dts)}')
    return dts


def run():
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.data_file:
        with open(args.config) as f:
            process_cfg = yaml.safe_load(f)['post_process']
        data_file = args.data_file
        if not args.out_file:
            raise ValueError('out_file must be path')
        out_file = args.out_file
    else:
        config = Config.parse(args.config)
        process_cfg = config.post_process
        data_file = str(config.output_dir / 'generated.csv')
        out_file = str(config.output_dir / 'processed.csv')
        config.setup_logger()

    logger.info('Start post processing')

    if process_cfg:

        dts = load_dataset(
            'csv',
            data_files=[data_file],
            split='train'
        )
        dts = post_process(config.post_process, dts)

        if config.answer_generations:
            dts = generate_answers(
                config=config.answer_generations,
                dataset=dts
            )
            if config.ag_filters:
                dts = filter_questions(dts, config.ag_filters)

        logger.info(f'Length: {len(dts)}')
        logger.info(f'Result columns: {dts.column_names}')
        dts.to_csv(out_file)


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        logger.error(e, exc_info=True)
        raise
