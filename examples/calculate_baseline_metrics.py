from argparse import ArgumentParser
import codecs
import json
import random
import os
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLM
import numpy as np
from transformers import AutoConfig, GenerationConfig, AutoTokenizer
import torch

from rag_bench import baseline, data, evaluator, results


RANDOM_SEED: int = 42
HIST_PRIVATE_QA_REPO_ID: str = 'ai-forever/hist-rag-bench-private-qa'
HIST_PRIVATE_TEXTS_REPO_ID: str = 'ai-forever/hist-rag-bench-private-texts'

def get_private_qa_dataset(version):
    return load_dataset(HIST_PRIVATE_QA_REPO_ID, revision=version)


def get_private_texts_dataset(version):
    return load_dataset(HIST_PRIVATE_TEXTS_REPO_ID, revision=version)


def get_public_to_private_texts_mapping(version):
    private_texts_ds = get_private_texts_dataset(version)
    mapping = {}
    for item in private_texts_ds['train']:
        mapping[item['public_id']] = item['id']
    return mapping


def auto_set_max_batched_tokens(model_path, gpu_memory_utilization=0.9, safety_margin=0.1):
    total_vram = torch.cuda.get_device_properties(0).total_memory

    config = AutoConfig.from_pretrained(model_path)
    model_memory = torch.cuda.memory_allocated()

    available_kv_memory = gpu_memory_utilization * total_vram - model_memory

    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    dtype_size = 2  # for bfloat16/fp16
    per_token_kv_memory = 2 * num_layers * hidden_size * dtype_size

    max_tokens = available_kv_memory // per_token_kv_memory
    max_tokens = int(max_tokens * (1 - safety_margin))

    block_size = 16
    max_tokens = (max_tokens // block_size) * block_size

    max_batched_tokens = 8192
    while max_batched_tokens <= max_tokens:
        max_batched_tokens *= 2
    max_batched_tokens //= 2

    return max_batched_tokens


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', type=str, required=True,
                        help='The used LLM.')
    parser.add_argument('-e', '--emb', dest='embedder', type=str, required=True,
                        help='The used embedder.')
    parser.add_argument('--doc', dest='doc_prompt', type=str, required=False, default=None,
                        help='The additional instruction for document embedding.')
    parser.add_argument('--query', dest='query_prompt', type=str, required=False, default=None,
                        help='The additional instruction for query embedding.')
    parser.add_argument('-o', '--out', dest='output_dir', type=str, required=True,
                        help='The output directory.')
    parser.add_argument('--top_k', dest='top_k', type=int, required=False, default=5,
                        help='The number of top items to retrieve.')
    parser.add_argument('--chunk_size', dest='chunk_size', type=int, required=False, default=1000,
                        help='The maximum size in characters of a single chunk.')
    parser.add_argument('--chunk_overlap', dest='chunk_overlap', type=int, required=False, default=100,
                        help='The overlap in characters between chunks.')
    parser.add_argument('--gpu', dest='gpu_memory_utilization', type=float, required=False, default=0.75,
                        help='The ratio (between 0 and 1) of GPU memory to reserve for the LLM weights, '
                             'activations, and KV cache in the vLLM library.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Should a more detailed log be printed?')
    args = parser.parse_args()

    output_dir = os.path.normpath(args.output_dir)
    if not os.path.isdir(output_dir):
        if output_dir == '':
            raise IOError('The output directory name is empty!')
        basedir = os.path.dirname(output_dir)
        if len(basedir) > 0:
            if not os.path.dirname(basedir):
                raise IOError(f'The directory "{basedir}" does not exist!')
        os.mkdir(output_dir)
    current_datetime = time.strftime('%Y%m%d%H%M%S', time.gmtime())
    printed_model_name = args.model.replace('/', '-').replace('\\', '-')
    printed_embedder_name = args.embedder.replace('/', '-').replace('\\', '-')
    answers_fname = os.path.join(
        output_dir,
        'baseline_' + printed_model_name + '_' + printed_embedder_name + '_' + current_datetime + '_answers.json'
    )
    evaluation_fname = os.path.join(
        output_dir,
        'baseline_' + printed_model_name + '_' + printed_embedder_name + '_' + current_datetime + '_eval.json'
    )

    # get public datasets (history ones)
    texts_ds, questions_ds, version = data.get_datasets(is_hist=True)

    # get private datasets (history ones)
    qa_dataset = get_private_qa_dataset(version)
    text_mapping = get_public_to_private_texts_mapping(version)

    llm_system_prompt = ('Внимательно проанализируйте заданный контекст и ответьте на вопрос пользователя '
                         'с использованием сведений, предоставленных в этом контексте.\n'
                         'Не давайте никаких объяснений и пояснений к своему ответу. Не пишите ничего лишнего. '
                         'Не извиняйтесь, не стройте диалог. Выдавайте только ответ и ничего больше.\n'
                         'Отвечайте на русском языке.\nЕсли в заданном контексте нет информации для ответа на '
                         'вопрос пользователя, то ничего не придумывайте и просто откажитесь отвечать.')
    llm_config = AutoConfig.from_pretrained(args.model)
    gen_config = GenerationConfig.from_pretrained(args.model)
    max_new_tokens = (512 if gen_config.max_new_tokens is None else gen_config.max_new_tokens)
    max_model_length = round(0.5 * (args.chunk_size * args.top_k)) + len(llm_system_prompt) + max_new_tokens
    if max_model_length > llm_config.max_position_embeddings:
        max_top_k = 2 * (llm_config.max_position_embeddings - max_new_tokens - len(llm_system_prompt)) // args.chunk_size
        err_msg = f'The number of top items to retrieve is too large! Expected less than {max_top_k}, got {args.top_k}.'
        raise ValueError(err_msg)
    print(f'{args.model} max_model_len = {max_model_length}')
    max_num_batched_tokens = auto_set_max_batched_tokens(args.model, args.gpu_memory_utilization)
    if max_num_batched_tokens < max_model_length:
        err_msg = (f'The max_model_length is too large. '
                   f'Expected less than {max_num_batched_tokens}, got {max_model_length}. '
                   f'Since max_model_len is calculated automatically based on top_k, then decrease top_k.')
        raise ValueError(err_msg)
    max_num_batched_tokens_ = 4
    while max_num_batched_tokens_ < max_model_length:
        max_num_batched_tokens_ *= 4
    max_num_batched_tokens = min(max_num_batched_tokens, max_num_batched_tokens_)
    print(f'{args.model} max_num_batched_tokens = {max_num_batched_tokens}')
    llm = VLLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_new_tokens=max_new_tokens,
        top_p=(0.95 if gen_config.top_p is None else gen_config.top_p),
        temperature=(0.7 if gen_config.temperature is None else gen_config.temperature),
        top_k=(20 if gen_config.top_k is None else gen_config.top_k),
        vllm_kwargs={
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'max_num_batched_tokens': max_num_batched_tokens,
            'max_model_len': max_model_length,
            'seed': RANDOM_SEED
        }
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    del gen_config

    text_emb_kwargs = dict()
    if args.doc_prompt is not None:
        text_emb_kwargs['prompt'] = args.doc_prompt
    query_emb_kwargs = dict()
    if args.query_prompt is not None:
        query_emb_kwargs['prompt'] = args.query_prompt
    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedder,
        model_kwargs={'trust_remote_code': True},
        encode_kwargs=text_emb_kwargs,
        query_encode_kwargs=query_emb_kwargs
    )
    if (args.doc_prompt is None) and (args.query_prompt is None):
        info_msg = 'The embeddings model will be used without any prompt.'
    elif (args.doc_prompt is not None) and (args.query_prompt is not None):
        info_msg = (f'The embeddings model will be used with the query prompt = "{query_emb_kwargs["prompt"]}" and '
                    f'with the document prompt = "{text_emb_kwargs["prompt"]}".')
    elif args.doc_prompt is not None:
        info_msg = f'The embeddings model will be used with the document prompt = "{text_emb_kwargs["prompt"]}".'
    else:
        info_msg = f'The embeddings model will be used with the query prompt = "{query_emb_kwargs["prompt"]}".'
    print(info_msg)

    retrieval = baseline.init_retriever(
        texts_ds,
        embedding_model,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    generation = baseline.init_generation(retrieval, llm, tok, system_prompt=llm_system_prompt)

    res = baseline.get_results(generation, questions_ds, write_logs=args.verbose)

    # saving the baseline results
    results.save(res, answers_fname)
    print(f'The test answers are successfully saved into "{answers_fname}".')

    evaluation_results = evaluator.evaluate_rag_results(res, qa_dataset, text_mapping)

    report_data = {
        'llm': args.model,
        'embedder': args.embedder,
        'top_k': args.top_k,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'DRAGON_version': version,
        'overall_results': evaluation_results.average_metrics['overall'],
        'results_by_category': dict(map(
            lambda k: (k, evaluation_results.average_metrics[k]),
            sorted(list(set(evaluation_results.average_metrics.keys()) - {'overall'}))
        ))
    }
    with codecs.open(evaluation_fname, mode='w', encoding='utf-8') as fp:
        json.dump(obj=report_data, fp=fp, ensure_ascii=False, indent=4)
    print(f'The evaluation report is successfully saved into "{evaluation_fname}".')


if __name__ == '__main__':
    main()
