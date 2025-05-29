import yaml
import logging
import pandas as pd

from dataclasses import dataclass, field
from datasets import IterableDataset
from datetime import datetime
from pandas import DataFrame
from pathlib import Path

from src.core.ModelLoader import LMConfig
from src.lm.prompts import load_prompt


class Config:
    def __init__(self,
                 format,
                 data_files,
                 raw_data_files,
                 output_dir,
                 process_columns,
                 syntax_config,
                 graph_extraction=None,
                 graph_questions=[],
                 answer_generations=[],
                 ag_filters=[],
                 pre_process=False,
                 post_process=None,
                 cache_path=None,
                 global_lm_config=None,
                 judge_lm_config=None,
                 judge_instructions=None,
                 filter_config=None):
        self.format = format
        self.data_files = data_files
        self.raw_data_files = raw_data_files
        self.output_dir = output_dir
        self.columns = process_columns

        self.global_lm_config = global_lm_config
        self.syntax_config = syntax_config
        self.graph_extraction = graph_extraction
        self.graph_questions = graph_questions
        self.answer_generations = answer_generations
        self.ag_filters = ag_filters
        self.judge_lm_config = judge_lm_config
        self.judge_instructions = judge_instructions
        self.filter_config = filter_config

        self.pre_process = pre_process
        self.post_process = post_process
        self.cache_path = cache_path
        self.gen_cache_file = str(cache_path / 'generation.arrow')

    def save2path(self, dataset, save_path, ext='csv'):
        if isinstance(dataset, IterableDataset):
            raise NotImplementedError('Iterable dataset saving is not ready')
        if isinstance(dataset, DataFrame):
            if ext == 'json':
                dataset.to_json(
                    save_path,
                    orient='records',
                    index=False
                )
            elif ext == 'csv':
                dataset.to_csv(
                    save_path,
                    index=False
                )
        else:
            if ext == 'json':
                dataset.to_json(
                    save_path
                )
            elif ext == 'csv':
                dataset.to_csv(
                    save_path
                )

    def save(self, dataset, name, ext='csv'):
        save_path = self.output_dir / f'{name}.{ext}'
        self.save2path(dataset, save_path, ext)

    def load_news_df(self):
        texts = []
        for rdf in self.data_files:
            if self.format == 'json':
                text = pd.read_json(rdf)[self.columns]
            elif self.format == 'csv':
                text = pd.read_csv(rdf)[self.columns]
            else:
                raise ValueError('Dataset must be csv or json')
            texts.append(text)
        texts = pd.concat(texts)
        return texts

    def setup_logger(self, init=False):
        log_dir = self.output_dir / 'logs'
        files = (
            sorted(list(log_dir.iterdir()))
            if log_dir.is_dir()
            else []
        )
        if not init and len(files) > 0:
            log_file = files[-1]
        else:
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / datetime.now().strftime(
                '%Y-%m-%d_%H-%M-%S.log'
            )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        file = logging.FileHandler(
            log_file,
            mode='a'
        )
        file.setLevel(logging.INFO)
        handlers = [console, file]
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%m-%d %H:%M',
            handlers=handlers
        )

    @classmethod
    def parse_data(cls, _cfg, output_dir):
        data_format = _cfg.get('format', 'json')
        data_dir = Path(_cfg['data_dir'])
        if 'dates' in _cfg:
            dates = _cfg['dates']
            raw_data_files = []
            for dpth in data_dir.iterdir():
                if dpth.name in dates:
                    raw_data_files += list(map(
                        str, dpth.glob(_cfg['file_pattern'])
                    ))
        else:
            raw_data_files = list(map(
                str, data_dir.glob(_cfg['file_pattern'])
            ))

        if 'pre_process' in _cfg:
            data_files = [
                str(output_dir / f'raw_data.{data_format}')
            ]
        else:
            data_files = raw_data_files

        return data_format, data_files, raw_data_files

    @classmethod
    def parse(cls, config_path):
        with open(config_path) as f:
            _cfg = yaml.safe_load(f)

        output_dir = Path(_cfg['output_dir'])
        data_format, data_files, raw_data_files = cls.parse_data(
            _cfg, output_dir
        )

        if not output_dir.is_dir():
            raise ValueError('`output_dir` must be directory')

        if 'cache_path' in _cfg:
            cache_path = Path(_cfg['cache_path'])
        else:
            cache_path = Path('./cache')

        global_lm_config = None
        if 'global_lm_config' in _cfg:
            global_lm_config = LMConfig.parse_yaml(_cfg['global_lm_config'])
        syntax_config = None
        if 'syntax_generation' in _cfg:
            syntax_config = SyntaxConfig.parse(_cfg['syntax_generation'])

        graph_extraction = None
        if 'graph_extraction' in _cfg:
            graph_extraction = GEConfig.parse(
                lm_config=_cfg['graph_extraction']['lm_config'],
                lm_config2=_cfg['graph_extraction']['lm_config2'],
                data_format=data_format,
                data_files=data_files,
                output_dir=output_dir,
                relation_mapping=_cfg['graph_extraction']['relation_mapping'],
                entity_mapping=_cfg['graph_extraction']['entity_mapping'],
                relation_index=_cfg['graph_extraction']['relation_index'],
                entity_index=_cfg['graph_extraction']['entity_index'],
                rebuild=_cfg['graph_extraction'].get('rebuild', True)
            )

        graph_questions = []
        if 'graph_questions' in _cfg:
            gq = _cfg['graph_questions']
            _lm_config = gq.get('lm_config', None)
            for _gqg in gq['tasks']:
                lm_config = _gqg.get('lm_config', None) or _lm_config
                gqg = GQGConfig.parse(
                    _gqg,
                    lm_config,
                    output_dir
                )
                graph_questions.append(gqg)

        answer_generations = []
        ag_filters = []
        for post_process_cfg in _cfg.get('post_process', []):
            if post_process_cfg['type'] == 'answer_generation':
                gen_ans_columns = [
                    m['column_name'] for m in post_process_cfg['models']
                ]
                for filtr in post_process_cfg.get('filters', []):
                    flt = AGFilterConfig.parse(
                        filtr,
                        columns=gen_ans_columns,
                    )
                    ag_filters.append(flt)
                if 'final_res_dir' in post_process_cfg:
                    final_res_dir = post_process_cfg['final_res_dir']
                else:
                    final_res_dir = output_dir
                for _model in post_process_cfg['models']:
                    column_name = _model['column_name']
                    lm_config = _model.get('lm_config', None) or _lm_config
                    mag = AGConfig.parse(lm_config, final_res_dir, column_name)
                    answer_generations.append(mag)
            if post_process_cfg.get('name') == 'match_ner':
                if 'ner_path' not in post_process_cfg:
                    post_process_cfg['ner_path'] = output_dir / 'ners.csv'

        judge_instructions = dict()
        judge_lm_config = None
        if 'judge' in _cfg:
            judge_dir = output_dir / 'judge'
            judge_dir.mkdir(exist_ok=True)
            judge_cfg = _cfg['judge']
            lm_config_path = judge_cfg['lm_config']
            judge_lm_config = LMConfig.parse_yaml(lm_config_path)
            for name, instr in judge_cfg['instructions'].items():
                lm_config = LMConfig.parse_yaml(
                    lm_config_path
                )
                lm_config.output_path = judge_dir / f'{name}.csv'
                lm_config.data_files = [str(output_dir / 'processed.csv')]
                lm_config.data_format = 'csv'
                lm_config.instruction = load_prompt(instr)
                judge_instructions[name] = lm_config

        return cls(
            format=data_format,
            data_files=data_files,
            raw_data_files=raw_data_files,
            output_dir=output_dir,
            process_columns=_cfg.get('process_columns', None),
            cache_path=cache_path,
            pre_process=_cfg.get('pre_process', []),
            post_process=_cfg.get('post_process', []),
            syntax_config=syntax_config,
            graph_extraction=graph_extraction,
            graph_questions=graph_questions,
            answer_generations=answer_generations,
            ag_filters=ag_filters,
            global_lm_config=global_lm_config,
            judge_lm_config=judge_lm_config,
            judge_instructions=judge_instructions,
            filter_config=_cfg.get('filter', []),
        )


@dataclass
class SyntaxConfig:
    run: bool = field(default=False)
    qtypes: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def parse(cls, _cfg):
        qtypes = {
            qt['type']: 'all' in qt
            for qt in _cfg.get('question_types', [])
        }
        return cls(True, qtypes)


@dataclass
class GEConfig:
    lm_config: LMConfig
    lm_config2: LMConfig
    graph_dir: Path
    st1_path: Path
    st2_path: Path
    st3_path: Path
    graph_path: Path
    ents_path: Path
    filtered_graph_path: Path
    filtered_ents_path: Path
    wikidata_graph_path: Path
    wikidata_ents_path: Path
    relation_mapping: Path = field(default=None)
    entity_mapping: Path = field(default=None)
    relation_index: Path = field(default=None)
    entity_index: Path = field(default=None)
    rebuild: bool = field(default=False)

    @classmethod
    def parse(cls,
              lm_config,
              lm_config2,
              data_format,
              data_files,
              output_dir,
              relation_mapping=None,
              entity_mapping=None,
              relation_index=None,
              entity_index=None,
              rebuild=False):
        graph_dir = output_dir / 'graph'
        graph_dir.mkdir(exist_ok=True)
        st1_path = str(graph_dir / 'stage1.json')
        st2_path = str(graph_dir / 'stage2.json')
        st3_path = str(graph_dir / 'stage3.json')

        lm_config = LMConfig.parse_yaml(lm_config)
        lm_config.data_format = data_format
        lm_config.data_files = data_files
        lm_config.output_path = st1_path

        lm_config2 = LMConfig.parse_yaml(lm_config2)
        lm_config2.data_format = 'json'
        lm_config2.data_files = [st2_path]
        lm_config2.output_path = st3_path
        return cls(
            lm_config=lm_config,
            lm_config2=lm_config2,
            graph_dir=graph_dir,
            st1_path=st1_path,
            st2_path=st2_path,
            st3_path=st3_path,
            graph_path=graph_dir / 'graph.ttl',
            ents_path=graph_dir / 'ents.csv',
            filtered_graph_path=graph_dir / 'filtered_graph.ttl',
            filtered_ents_path=graph_dir / 'filtered_ents.csv',
            wikidata_graph_path=graph_dir / 'wikidata_graph.ttl',
            wikidata_ents_path=graph_dir / 'wikidata_ents.csv',
            relation_mapping=relation_mapping,
            entity_mapping=entity_mapping,
            relation_index=relation_index,
            entity_index=entity_index,
            rebuild=rebuild
        )


@dataclass
class GQGConfig:
    type: str
    n: int
    lm_config: LMConfig
    rebuild_task: bool
    task_path: Path
    raw_path: Path
    result_path: Path

    @classmethod
    def parse(cls,
              pre_conf,
              lm_config_path,
              output_dir):
        type = pre_conf['type']
        task_path = output_dir / 'tasks' / f'{type}.csv'
        task_path.parent.mkdir(exist_ok=True)
        raw_path = output_dir / 'raw_questions' / f'{type}.json'
        raw_path.parent.mkdir(exist_ok=True)
        result_path = output_dir / 'result' / f'{type}.csv'
        result_path.parent.mkdir(exist_ok=True)

        with open(lm_config_path) as f:
            _cfg = yaml.safe_load(f)

        _cfg['data_format'] = 'csv'
        _cfg['data_files'] = [str(task_path)]
        _cfg['output_path'] = raw_path
        if 'override' in pre_conf:
            _cfg.update(pre_conf['override'])
        lm_config = LMConfig.parse(_cfg)

        return cls(
            type=type,
            n=pre_conf.get('n', 50),
            lm_config=lm_config,
            rebuild_task=pre_conf.get('rebuild_task', True),
            task_path=task_path,
            raw_path=raw_path,
            result_path=result_path
        )


@dataclass
class AGConfig:
    lm_config: LMConfig
    final_res_dir: str
    column_name: str

    @classmethod
    def parse(cls,
              lm_config_path,
              final_res_dir,
              column_name):

        with open(lm_config_path) as f:
            _cfg = yaml.safe_load(f)
        lm_config = LMConfig.parse(_cfg)

        return cls(
            lm_config=lm_config,
            final_res_dir=final_res_dir,
            column_name=column_name
        )


@dataclass
class AGFilterConfig:
    type: str
    model_name: str
    thr: int
    columns: list
    drop: bool

    @classmethod
    def parse(cls, filtr, columns):
        return cls(
            type=filtr.get('type'),
            model_name=filtr.get('model_name'),
            thr=float(filtr.get('thr')),
            columns=columns,
            drop=True if 'true' in str(filtr.get('drop')).lower() else False
        )
