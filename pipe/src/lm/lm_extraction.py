import json
import logging

from argparse import ArgumentParser
from openai import OpenAI

from src.core.ModelLoader import (
    ModelLoader,
    vLLM_ModelLoader,
    LMConfig
)
from src.core.AskLM import (
    AnswerGenerator,
    vLLM_AnswerGenerator,
    API_AnswerGenerator
)

logger = logging.getLogger(__name__)


def extract(config, model=None, tokenizer=None, dataset=None, write=True):
    if dataset is None:
        dataset = config.load_dataset()

    if config.model_type == 'api':
        api = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
        )

        answer_generator = API_AnswerGenerator(
            api=api,
            model_name=config.model_path,
            dataset=dataset,
            instruction=config.instruction,
            max_context_length=config.max_context_length,
            max_new_tokens=config.max_new_tokens,
            chat_model=config.chat_model,
            sys_prompt=config.sys_prompt,
            few_shots=config.few_shots,
            prompt_subs=config.prompt_subs
        )
    else:
        if model is None or tokenizer is None:
            if config.model_type == 'vllm':
                model_loader = vLLM_ModelLoader(
                    model_path=config.model_path,
                    model_torch_dtype=config.model_torch_dtype,
                    tensor_parallel_size=config.tp_size
                )
            else:
                model_loader = ModelLoader(
                    model_path=config.model_path,
                    model_torch_dtype=config.model_torch_dtype,
                    device=config.device
                )

            model, tokenizer = model_loader.model_load()

        Generator = (
            vLLM_AnswerGenerator
            if config.model_type == 'vllm'
            else AnswerGenerator
        )
        answer_generator = Generator(
            model=model,
            tokenizer=tokenizer,
            device=config.device,
            dataset=dataset,
            instruction=config.instruction,
            max_context_length=config.max_context_length,
            max_new_tokens=config.max_new_tokens,
            chat_model=config.chat_model,
            sys_prompt=config.sys_prompt,
            few_shots=config.few_shots,
            prompt_subs=config.prompt_subs,
            truncate=config.truncate
        )

    answers = answer_generator.generate_answers()
    if write:
        with open(config.output_path, 'w', encoding='utf-8') as outfile:
            json.dump(answers, outfile, ensure_ascii=False)
        logger.info(f"predictions were saved here: {config.output_path}")
    return answers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    config = LMConfig.parse_yaml(args.config)
    extract(config)
