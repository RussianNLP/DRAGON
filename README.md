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

### Client library

The client library, located in `lib/`, provides tools to benchmark RAG pipelines. It allows users to:

1.  **Load Datasets**: Easily fetch text and question datasets from HuggingFace using functions in `rag_bench.data`. It ensures version consistency between datasets.
    ```python
    from rag_bench.data import get_datasets

    texts_ds, questions_ds, version = get_datasets()
    ```

2.  **Build a RAG Pipeline**: The `rag_bench.baseline` module offers a reference implementation for a RAG pipeline.
    *   Initialize a retriever (e.g., ChromaDB with MMR search) from your text data and an embedding model:
        ```python
        from rag_bench.baseline import init_retriever
        retriever = init_retriever(texts_ds, embedding_model)
        ```
    *   Initialize a generation chain using the retriever, a language model, and an optional prompt:
        ```python
        from rag_bench.baseline import init_generation
        generation_chain = init_generation(retriever, llm)
        ```

3.  **Generate Results**: Use the configured pipeline to process questions from the dataset.
    ```python
    from rag_bench.baseline import get_results

    results = get_results(generation_chain, questions_ds)
    ```

4.  **Evaluate Performance**: The `rag_bench.evaluator` module helps assess the RAG pipeline's effectiveness.
    *   It calculates retrieval metrics like Hit Rate and MRR.
    *   It computes generation metrics such as ROUGE scores, Exact Match, and Substring Match.
    ```python
    from rag_bench.evaluator import evaluate_rag_results, RAGEvaluationResults

    evaluation_output = evaluate_rag_results(results, questions_ds)

    evaluation_output.to_table()
    ```

Key modules are:
*   `rag_bench.data`: For dataset loading.
*   `rag_bench.baseline`: For RAG pipeline construction and execution.
*   `rag_bench.evaluator`: For evaluating the pipeline's output.
*   `rag_bench.constants`: Stores repository IDs for datasets.
*   `rag_bench.helper`: Utility functions.


### QA dataset generation pipeline

<p align="center">
    <img src="./static/images/qg_pipeline.png" width="540px" />
</p>

The Data Generation pipeline consists of 2 stages: KG Extraction and Question Generation. The KG Extraction retrieves factual information from texts and preserves the most specific and fresh facts in form of a Knowledge Graph. The Question Generation module samples subgraphs of a certain structure to generate a question-answer pair with LLM.
