# kb on DRAGON (hist, version 1.15.0)

Third-party results for [kb](https://github.com/alterfo/kb), a self-contained graphRAG knowledge base written in Go (hybrid dense + BM25 retrieval, entity graph, Graph-of-Thoughts answer synthesis; one SQLite file for vectors, graph and metadata).

These are self-run submissions scored with the official `rag_bench.evaluator.evaluate_rag_results`, not leaderboard entries produced by the hosted backend.

## Correction

The first version of this PR was labelled `qwen3.8`, but the `qwen3.8:latest` tag on my Ollama host was pointing to a gemma4 model. I apologize for the mistake. The original run is kept below under its real name (`gemma4`), and a new run on the actual qwen3.8 is added next to it. Both use the same kb build and settings; only the chat model differs, and each run has its own index built from scratch.

## Setup (both runs)

- Dataset: `ai-forever/hist-rag-bench-*`, revision `1.15.0`, 542 texts, 600 questions (150 per type).
- Embeddings: `qwen3-embedding:0.6b`.
- Index: all 542 texts indexed with entity/relation extraction and community summaries.
- Answering: retrieval and Graph-of-Thoughts synthesis, followed by one extra chat call that reduces the answer to its bare value (name, number, date or short list), so that Exact Match, Substring Match and ROUGE are meaningful. `found_ids` are the retrieved public text ids in ranked order.
- Chat / graph extraction model, thinking disabled, served by Ollama on local hardware, no external API calls:
  - `gemma4/`: gemma4, 25.2B, Q4_K_M (run of 2026-10-04, previously mislabelled `qwen3.8`).
  - `qwen3.8/`: qwen3.8, architecture qwen35, 27.3B, Q4_K_M (run of 2026-10-09).

## Results

### Leaderboard view

Same columns and averaging as the leaderboard app (per-category mean of `simple/set/mh/cond`, `Total Score` = mean of Retrieval and Generation). These entries have no `precision` (the official evaluator does not compute it), so Retrieval (avg) is the mean of hit rate and MRR only and is not directly comparable with entries that include precision. The other leaderboard entries are on version 1.11.0.

| Model | Embeddings | Top k | Retrieval (avg) | Generation (avg) | Total Score | Version | Last Updated |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (kb graphRAG) (aa045f) | qwen3-embedding:0.6b | 10 | 0.7716 | 0.6474 | 0.7095 | 1.15.0 | 2026-10-09 |
| gemma4-25B (kb graphRAG) (fe39ea) | qwen3-embedding:0.6b | 10 | 0.7505 | 0.6244 | 0.6874 | 1.15.0 | 2026-10-04 |

### qwen3.8 (`qwen3.8/`)

| Metric | Overall | cond | mh | set | simple |
|---|---|---|---|---|---|
| Hit Rate | 0.9153 | 0.9167 | 0.8911 | 0.9333 | 0.9200 |
| MRR | 0.6279 | 0.5849 | 0.6147 | 0.6944 | 0.6175 |
| ROUGE-1 | 0.6660 | 0.8315 | 0.6305 | 0.5754 | 0.6265 |
| ROUGE-2 | 0.4496 | 0.6117 | 0.3559 | 0.3547 | 0.4761 |
| ROUGE-L | 0.6289 | 0.8255 | 0.6296 | 0.4338 | 0.6265 |
| Exact Match | 0.3550 | 0.7067 | 0.3667 | 0.0000 | 0.3467 |
| Substring Match | 0.4300 | 0.7400 | 0.4600 | 0.0467 | 0.4733 |

### gemma4 (`gemma4/`)

| Metric | Overall | cond | mh | set | simple |
|---|---|---|---|---|---|
| Hit Rate | 0.8975 | 0.9167 | 0.8878 | 0.9056 | 0.8800 |
| MRR | 0.6035 | 0.6122 | 0.6031 | 0.6037 | 0.5950 |
| ROUGE-1 | 0.6441 | 0.7630 | 0.6391 | 0.5605 | 0.6137 |
| ROUGE-2 | 0.4319 | 0.5598 | 0.3838 | 0.3259 | 0.4583 |
| ROUGE-L | 0.6048 | 0.7597 | 0.6357 | 0.4131 | 0.6107 |
| Exact Match | 0.3017 | 0.5667 | 0.4067 | 0.0000 | 0.2333 |
| Substring Match | 0.3500 | 0.5933 | 0.4333 | 0.0067 | 0.3667 |

## Files

Per run directory:

- `submission.json`: `{public_id: {found_ids, model_answer}}`, exactly the structure `evaluate_rag_results` expects.
- `official-report.json`: `average_metrics` returned by the evaluator, same numbers as above.

## Reproduce the scoring

```python
import json
from datasets import load_dataset
from rag_bench import evaluator

version = "1.15.0"
results = {k: {"found_ids": [int(x) for x in v["found_ids"]], "model_answer": v["model_answer"]}
           for k, v in json.load(open("qwen3.8/submission.json")).items()}
qa = load_dataset("ai-forever/hist-rag-bench-private-qa", revision=version)
texts = load_dataset("ai-forever/hist-rag-bench-private-texts", revision=version)
mapping = {t["public_id"]: t["id"] for t in texts["train"]}
print(evaluator.evaluate_rag_results(results, qa, mapping).average_metrics)
```

## Notes

- The `set` type scores near zero on Exact Match because list answers are compared as a whole string, so any difference from the gold list fails the match.
- The short-answer reduction step is part of the local kb build used for these runs and has not been published in the kb repository yet. The submission files and the scoring above can be verified independently of it.
