import logging
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from src.core.ModelLoader import LMConfig
from src.process.config import Config
from src.lm.utils import extract_score
from src.lm.lm_extraction import extract
from src.core.ModelLoader import (
    ModelLoader,
    vLLM_ModelLoader
)

logger = logging.getLogger('src.process.judge')


def judge(lm_config, inst_configs, output_file):
    logger.info('Judge evaluation')

    model = None
    tokenizer = None
    if lm_config.model_type != 'api':
        if lm_config.model_type == 'vllm':
            model_loader = vLLM_ModelLoader(
                model_path=lm_config.model_path,
                model_torch_dtype=lm_config.model_torch_dtype,
                tensor_parallel_size=lm_config.tp_size
            )
        elif lm_config.model_type == 'hf':
            model_loader = ModelLoader(
                model_path=lm_config.model_path,
                model_torch_dtype=lm_config.model_torch_dtype,
                device=lm_config.device
            )

        model, tokenizer = model_loader.model_load()

    res_files = dict()
    for name, inst_cfg in tqdm(inst_configs.items()):
        logger.info(f'Run {name} judge')
        res_files[name] = inst_cfg.output_path
        extract(inst_cfg, model, tokenizer)

    res_df = None
    for name, pth in res_files.items():
        df = pd.read_json(pth)
        if res_df is None:
            res_df = df.drop(columns='model_answer').copy()
        res_df[f'{name}_score'] = df['model_answer'].map(extract_score)
        res_df[f'{name}_desc'] = df['model_answer']
    if output_file is not None:
        logger.info(f'Judge scores saved to {output_file}')
        res_df.to_csv(str(output_file), index=False)


def run():
    parser = ArgumentParser()
    parser.add_argument('--global_config', type=Path)
    parser.add_argument('--lm_config', type=Path)
    parser.add_argument('--inst_configs', type=str)
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--data_file', type=str)
    args = parser.parse_args()

    if args.global_config:
        print('parse global')
        cfg = Config.parse(args.global_config)
        lm_config = cfg.judge_lm_config
        inst_configs = cfg.judge_instructions
        out_file = cfg.output_dir / 'judges.csv'
        cfg.setup_logger()

    elif args.lm_config and args.inst_configs:
        lm_config = LMConfig.parse_yaml(args.lm_config)

        inst_configs = dict()
        for p in args.inst_configs.split(','):
            p = Path(p)
            lmcfg = LMConfig.parse_yaml(p)
            if args.data_file:
                lmcfg.data_files = [args.data_file]
            if args.output_dir:
                default_out = Path(lmcfg.output_path)
                filename = default_out.name
                lmcfg.output_path = str(args.output_dir / filename)
            inst_configs[p.stem] = lmcfg

        out_file = args.output_file
    else:
        logger.error('You must set global_config or lm_config')
        raise ValueError('You must set global_config or lm_config')

    judge(lm_config, inst_configs, out_file)


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        logger.error(e, exc_info=True)
        raise
