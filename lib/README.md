# Baseline and evaluation pipeline for Dynamic RAG On News benchmark (DRAGON)

## Baseline (reference implementation)

The module `baseline.py` provides a simple **vector-based RAG baseline** used in DRAGON / `rag-bench` to build a retriever over the dataset and run a retrieval-augmented generation chain to produce predictions that can be evaluated with retrieval and generation metrics.

### What the module does

- **Indexes the dataset** into a vector store (Chroma) using a provided embedding model.
- **Retrieves top‑k relevant chunks** for each question (MMR retrieval).
- **Generates an answer strictly from retrieved context** using an LLM + prompt template.
- **Formats outputs** into a compact structure expected by the evaluation code:
  - `found_ids`: IDs of retrieved public documents
  - `model_answer`: model-generated answer text

### Main functions

#### The function `init_retriever(dataset, embedding_model, top_k=5, chunk_size=500, chunk_overlap=100, batch_size=1000)`
Creates a retriever backed by **Chroma**:

1. Converts each `dataset["train"]` item into a `langchain_core.documents.Document`:
   - `page_content = item["text"]`
   - `metadata = {"id": item["id"]}` (used later for retrieval metrics)
2. Splits documents into overlapping chunks using `RecursiveCharacterTextSplitter`.
3. Resets the Chroma collection and **adds chunks in batches** (controlled by `batch_size`).
4. Exposes a retriever via `vector_store.as_retriever(...)` with:
   - `search_type="mmr"`
   - `k=top_k`

Returns: a LangChain retriever object.

#### The function `init_generation(retriever, model, tokenizer, system_prompt="")`
Builds the generation chain:

- Uses a **system prompt**. If none is provided, a default instruction (in Russian) is used:
  - *Отвечайте на вопрос строго по контексту. Ответ — одной строкой. Если ответа нет в контексте, то напишите: не хватает данных в контексте.*
- Creates a chat-style prompt via `tokenizer.apply_chat_template(...)`.
- Combines:
  - `create_stuff_documents_chain(model, prompt)` (stuffs retrieved context into the prompt)
  - `create_retrieval_chain(retriever, document_chain)` (retrieval + generation)

Returns: a LangChain retrieval-generation chain that can be invoked with `{"input": question}`.

#### The function `get_results(generation_chain, dataset, skip=None, take=None, write_logs=False, sleep_time=0)`
Runs inference over `dataset["train"]`:

- Supports slicing via:
  - `skip` (start offset)
  - `take` (number of samples from the start)
- For each item, calls:
  - `generation_chain.invoke({"input": item["question"]})`
- Optional verbose printing (`write_logs`) and throttling (`sleep_time`).
- Converts raw LangChain responses into evaluation-ready format using `prepare_results()`.

Returns: a dictionary keyed by item ID with:
- `found_ids`: list of retrieved public document IDs (`metadata["id"]`)
- `model_answer`: generated answer string

#### The function `prepare_results(data)`
Post-processes LangChain outputs into the minimal schema expected by the evaluator.

### Logging
The module uses `log()` from `helper.py` to print messages and append them to a log file (`LOG_FILE`), including timestamps.

## Evaluation

The module `evaluator.py` implements the evaluation layer of `rag-bench` used in DRAGON. It computes **retrieval quality metrics** (Hit Rate, MRR) and **answer quality metrics** (ROUGE-1/2/L, Exact Match, Substring Match), aggregates them **overall** and **by question type**, and can render results as readable tables.

### Key classes

#### The class `RAGEvaluator`
Encapsulates metric computation.

- **Russian-aware tokenization for ROUGE**:
  - Uses `SnowballStemmer("russian")` and a custom `tokenize_ru()` to stem tokens.
  - Configures `rouge_score.rouge_scorer.RougeScorer` with this tokenizer.

- **Retrieval metrics**: the static method `evaluate_retrieval(retrieved_doc_ids, relevant_doc_ids)`:
  - **Hit Rate** with supporting cases where a question has **multiple relevant documents**.
  - **MRR (Mean Reciprocal Rank)**: finds the first retrieved document that is relevant and returns `1/(rank)`, else `0`.

- **Generation metrics**: the method `evaluate_generation(generated_answer, reference_answer)`:
  - `rouge1`, `rouge2`, `rougeL` (f-measure)
  - `exact_match`: normalized string equality (known as Exact Match, or EM)
  - `substring_match`: checks whether the generated answer appears inside the reference answer (case-insensitive)

Additionally:
- The method `normalize_text()` collapses whitespace for robust EM / substring checks.
- The evaluator can strip chain-of-thought artifacts if present (see `THINK_END_TOKEN` usage below).

#### The class `RAGEvaluationResults`
A small container for:
- `individual_results`: per-sample metrics (retrieval + generation + question type)
- `average_metrics`: aggregated metrics (overall + per question type)

Includes:
- `to_dict()` / `from_dict()` for serialization
- `to_table(overall_only=True)` to print and return tabulated summaries (uses `log()` from `helper.py`)

### Main entry point

#### `evaluate_rag_results(results, dataset, text_mapping)`
Evaluates model outputs against the (private) reference dataset.

**Inputs**
- `results`: model predictions indexed by question `public_id`, each item must contain:
  - `found_ids`: list of retrieved public document IDs (returned by the pipeline)
  - `model_answer`: generated answer text
- `dataset`: evaluation dataset (uses `dataset["train"]`), each sample provides:
  - `public_id`, `type`, `answer`, and `text_ids` (reference relevant documents)
- `text_mapping`: mapping required to align document IDs across dataset partitions (see below)

**Public vs private document IDs (why `text_mapping` matters)**  
DRAGON uses:
- a **public** texts dataset (used by baseline retrieval / indexing), and
- a **private** questions+references dataset where the *same texts* exist but have **different IDs**.

Because the baseline retrieves and returns **public text IDs**, while the private ground truth stores **private text IDs**, `evaluate_rag_results()` converts predicted IDs via:

```python
found_doc_ids = [int(text_mapping[public_id_]) for public_id_ in predicted["found_ids"]]
```

Without this mapping, retrieval metrics (Hit Rate / MRR) would be computed on mismatched ID spaces and become invalid.

**Multiple relevant documents per question**  
Ground-truth `text_ids` is parsed from a string (via `ast.literal_eval`) and may contain:
- a single ID,
- multiple IDs,
- nested lists of IDs.

The function flattens and deduplicates them into `relevant_doc_ids`, enabling correct computation of Hit Rate and MRR when **more than one document is considered relevant**.

**Answer post-processing (`THINK_END_TOKEN`)**
If `model_answer` contains `THINK_END_TOKEN` (from `constants.py`, default `</think>`), everything up to and including that token is removed before generation metrics are computed. This helps ignore potential “thinking” prefixes.

**Aggregation**
After computing per-sample metrics, the module aggregates mean scores:
- **Overall** across all samples
- **Per question type** (based on `sample["type"]`)

Returns: `RAGEvaluationResults(individual_results, average_metrics)`.

### Produced metrics

- **Retrieval**
  - `hit_rate`
  - `mrr`

- **Generation**
  - `rouge1`, `rouge2`, `rougeL`
  - `exact_match`
  - `substring_match`

### Logging
`to_table()` logs formatted metric tables via `helper.log()`, which writes to `LOG_FILE` (see `constants.py`).