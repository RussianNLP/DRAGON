# Dynamic RAG On News benchmark (DRAGON)

<p align="center">
  <img src="./static/images/title.png" width="260px" />
</p>

## Architecture

The benchmark is designed to evaluate retrieval-augmented generation (RAG) systems in a realistic way, dynamically evolving news domain. It's architecture prioritizes modularity, automation, and reproducibility while addressing the core challenges in the RAG evaluation landscape.

The whole pipeline of the benchmark architecture can be explored in the following diagram:

<p align="center">
    <img src="./static/images/dragon_pipeline.png" width="360px" />
</p>

### Datasets

Our QA datasets are available on Hugging Face:

- [ai-forever/hist-rag-bench-public-questions](https://huggingface.co/datasets/ai-forever/hist-rag-bench-public-questions)
- [ai-forever/hist-rag-bench-public-texts](https://huggingface.co/datasets/ai-forever/hist-rag-bench-public-texts)
- [ai-forever/hist-rag-bench-private-qa](https://huggingface.co/datasets/ai-forever/hist-rag-bench-private-qa)
- [ai-forever/hist-rag-bench-private-texts](https://huggingface.co/datasets/ai-forever/hist-rag-bench-private-texts)

Datasets are updated once a month. To get the list of version programmatically you can use the following snippet:

```python
from huggingface_hub import list_repo_refs

def get_ds_versions(repo_id):
    repo_refs = list_repo_refs(repo_id, repo_type="dataset")
    return [ref.name for ref in repo_refs.tags]

#['1.15.0',
# '1.14.0',
# '1.13.0',
# '1.12.0',
#  ...
#  '1.0.0'
#]
```

Version 1.15.0 refers to December 2025, 1.14.0 — to November 2025, etc.

You can load a specific version of dataset:

```python
from datasets import load_dataset

ds = load_dataset('ai-forever/hist-rag-bench-private-qa', revision="1.13.0")
```

### QA dataset generation pipeline

<p align="center">
    <img src="./static/images/qg_pipeline.png" width="540px" />
</p>

The Data Generation pipeline consists of 2 stages: KG Extraction and Question Generation. The KG Extraction retrieves factual information from texts and preserves the most specific and fresh facts in form of a Knowledge Graph. The Question Generation module samples subgraphs of a certain structure to generate a question-answer pair with LLM.

### Usage

You can see the example of evaluating the baseline in [calculate_baseline_metrics.ipynb](https://github.com/RussianNLP/DRAGON/blob/main/examples/calculate_baseline_metrics.ipynb).

## Citation

```
@misc{chernogorskii2025dragondynamicragbenchmark,
      title={DRAGON: Dynamic RAG Benchmark On News}, 
      author={Fedor Chernogorskii and Sergei Averkiev and Liliya Kudraleeva and Zaven Martirosian and Maria Tikhonova and Valentin Malykh and Alena Fenogenova},
      year={2025},
      eprint={2507.05713},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.05713}, 
}
```
