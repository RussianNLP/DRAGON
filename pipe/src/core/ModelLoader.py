import os
import yaml
import torch
import transformers

from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from vllm import LLM

from src.lm.prompts import load_prompt


class LMConfig:
    def __init__(self, config):
        self.model_type = config.get('model_type', 'hf')
        self.model_path = config['model_path']
        self.instruction = config['instruction']
        self.data_files = config['data_files']
        self.data_format = config['data_format']
        self.output_path = config['output_path']

        self.model_torch_dtype = config.get('model_torch_dtype', 'float16')
        self.tokenizer_path = config.get('tokenizer_path', self.model_path)
        self.device = config.get('device', 'cpu')
        self.chat_model = config.get('chat_model', False)
        self.max_new_tokens = config.get('max_new_tokens', 1000)
        self.sys_prompt = config.get('sys_prompt', None)
        self.few_shots = config.get('few_shots', None)
        self.max_context_length = config.get('max_context_length', 2048)
        self.prompt_subs = config.get('prompt_subs', None)
        self.tp_size = config.get('tp_size', 1)
        self.api_base = config.get('api_base', None)
        self.api_key = os.environ.get('API_KEY', 'EMPTY')
        self.truncate = config.get('truncate', None)

    @classmethod
    def parse(cls, _cfg):
        if 'data_files' not in _cfg:
            data_path = Path(_cfg['dataset_path'])
            if data_path.is_file():
                data_files = [str(data_path)]
            elif data_path.is_dir():
                data_files = list(map(str, data_path.iterdir()))
            else:
                data_files = []
            _cfg['data_files'] = data_files
            if data_files:
                _cfg['data_format'] = data_files[0].split('.')[-1]
            else:
                _cfg['data_format'] = 'json'

        config = cls(_cfg)
        if config.chat_model:
            config.sys_prompt = load_prompt(config.sys_prompt)
            config.few_shots = load_prompt(config.few_shots)
            config.instruction = load_prompt(config.instruction)
        else:
            config.instruction = load_prompt(config.instruction)
        return config

    @classmethod
    def parse_yaml(cls, config_path):
        with open(config_path) as f:
            _cfg = yaml.safe_load(f)
        return cls.parse(_cfg)

    def load_dataset(self):
        if self.data_format == 'csv':
            dataset = load_dataset(
                self.data_format,
                data_files=self.data_files,
                split='train',
                keep_default_na=False
            )
        else:
            dataset = load_dataset(
                self.data_format,
                data_files=self.data_files,
                split='train'
            )
        return dataset


class ModelLoader:
    def __init__(
        self, model_path, model_torch_dtype, tokenizer_path=None, device="cpu"
    ):
        self.model_path = model_path
        self.device = device
        self.model_torch_dtype = self.get_dtype(model_torch_dtype)
        if tokenizer_path:
            self.tokenizer_path = tokenizer_path
        else:
            self.tokenizer_path = model_path

    def get_dtype(self, dtype):
        dct_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float,
        }
        if dtype in dct_dtypes:
            return dct_dtypes[dtype]
        else:
            return torch.float

    def model_load(self):
        config = transformers.AutoConfig.from_pretrained(self.model_path)
        if "LlamaForCausalLM" in config.architectures:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                # attn_implementation="flash_attention_2",
                trust_remote_code=True,
                torch_dtype=self.model_torch_dtype,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=self.model_torch_dtype,
            ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )
        return model, tokenizer


class vLLM_ModelLoader(ModelLoader):
    def __init__(
        self,
        model_path,
        model_torch_dtype,
        tokenizer_path=None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        device="cpu",
    ):
        super().__init__(model_path, model_torch_dtype, tokenizer_path, device)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

    def model_load(self):
        model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.model_torch_dtype,
            trust_remote_code=True,
            distributed_executor_backend="ray",
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )

        return model, tokenizer
