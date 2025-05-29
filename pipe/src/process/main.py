import logging
import pandas as pd

from argparse import ArgumentParser
from datasets import Dataset

from src.process.config import Config
from src.core.ModelLoader import (
    ModelLoader,
    vLLM_ModelLoader
)
from src.process.pre_process import pre_process
from src.process.generate import (
    generate_syntax,
    extract_graph,
    generate_from_graph
)

logger = logging.getLogger('src.process.main')


def main(config: Config):
    if config.pre_process:
        pre_process(config)

    if config.syntax_config:
        dts = generate_syntax(config)
        logger.info(f'Number of questions: {len(dts)}')

    model = None
    tokenizer = None
    if config.global_lm_config:
        logger.info('Use single model')
        if config.global_lm_config.model_type == 'vllm':
            model_loader = vLLM_ModelLoader(
                model_path=config.global_lm_config.model_path,
                model_torch_dtype=config.global_lm_config.model_torch_dtype,
                tensor_parallel_size=config.global_lm_config.tp_size
            )
        elif config.global_lm_config.model_type == 'hf':
            model_loader = ModelLoader(
                model_path=config.global_lm_config.model_path,
                model_torch_dtype=config.global_lm_config.model_torch_dtype,
                device=config.global_lm_config.device
            )

        model, tokenizer = model_loader.model_load()

    if config.graph_extraction:
        graph, ents = extract_graph(
            config.graph_extraction,
            model,
            tokenizer
        )

    if config.graph_questions:
        news_df = config.load_news_df()
        dfs = []
        for cfg in config.graph_questions:
            df = generate_from_graph(
                cfg,
                graph,
                ents,
                model,
                tokenizer,
                news_df=news_df
            )
            if df is not None:
                df['type'] = cfg.type
                dfs.append(df)
        try:
            df = pd.concat(dfs).reset_index(drop=True)
            df['answer'] = df['answer'].map(str)

            dts = Dataset.from_pandas(df, preserve_index=False)
        except ValueError as e:
            logger.error('No questions generated.' +
                         'The cause is')
            raise e

    if dts:
        logger.info(f'Length: {len(dts)}')
        logger.info(f'Result columns: {dts.column_names}')
        config.save(dts, name='generated')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    config = Config.parse(args.config)
    config.setup_logger(init=True)
    try:
        main(config)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise
