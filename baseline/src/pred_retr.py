import json
import yaml

from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from rag_bench.baseline import init_retriever
from rag_bench.helper import get_ds_versions


def retrieve_predict(embs_model,
                     output_path,
                     qa_dataset,
                     text_dataset,
                     top_k=20,
                     chunk_size=500,
                     chunk_overlap=100,
                     batch_size=1000
                     ):
    retriever = init_retriever(
        text_dataset,
        embs_model,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size
    )
    
    results = []
    for question in qa_dataset['train']:
        res = retriever.invoke(question['question'])
        res = {
            question['id']: [
                doc.metadata['id'] for doc in res
            ]
        }
        results.append(res)

    with open(output_path, "w") as f:
        json.dump(results, f)
    

def main(config, version, output_dir, cache_dir, pub_texts, pub_questions):
    texts_ds = load_dataset(
        pub_texts,
        revision=version,
        cache_dir=cache_dir
    )
    questions_ds = load_dataset(
        pub_questions,
        revision=version,
        cache_dir=cache_dir
    )

    for model_cfg in config['models']:
        print('MODEL:', model_cfg['path'])
        # embs_model = PromptedHFEmbeddings(
        #     model_name=model_cfg['path'],
        #     q_instr=model_cfg['query_instr'],
        #     text_instr=model_cfg['text_instr']
        # )

        text_kwargs, q_kwargs = dict(), dict()
        if 'query_instr' in model_cfg:
            q_kwargs['prompt'] = model_cfg['query_instr']
        if 'text_instr' in model_cfg:
            text_kwargs['prompt'] = model_cfg['text_instr']

        embs_model = HuggingFaceEmbeddings(
            model_name=model_cfg['path'],
            encode_kwargs=text_kwargs,
            query_encode_kwargs=q_kwargs
        )
        name = model_cfg['path'].split('/')[-1]
        for i, cfg in enumerate(config['storage_params']):
            output_path = output_dir / f'{name}_{i}.json'
            retrieve_predict(
                embs_model,
                output_path,
                questions_ds,
                texts_ds,
                **cfg
            )


if __name__ == "__main__":
    # versions = get_ds_versions(_pub_texts)
    # print(versions)
    # versions = get_ds_versions(_pub_questions)
    # print(versions)

    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--pub_texts', type=str, default="ai-forever/test-rag-bench-public-texts")
    parser.add_argument('--pub_questions', type=str, default="ai-forever/test-rag-bench-public-questions")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir / args.version / 'retrievals'
    output_dir.mkdir(parents=True, exist_ok=True)

    main(config, args.version, output_dir, args.cache_dir, args.pub_texts, args.pub_questions)
