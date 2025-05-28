from typing import Any, Dict

import numpy as np
from rouge_score import rouge_scorer
from tabulate import tabulate
from nltk.stem.snowball import SnowballStemmer
from types import SimpleNamespace
import re


from .helper import log


class RAGEvaluator:
    def __init__(self):
        self.rus_stem = SnowballStemmer("russian")

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"],
            tokenizer=SimpleNamespace(tokenize=self.tokenize_ru),
            use_stemmer=False
        )

    def tokenize_ru(self, text: str):
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return [self.rus_stem.stem(t) for t in tokens]

    def evaluate_retrieval(self, retrieved_doc_ids, relevant_doc_id):
        metrics = {}

        # hit rate
        metrics["hit_rate"] = 1.0 if relevant_doc_id in retrieved_doc_ids else 0.0

        # mrr
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id == relevant_doc_id:
                metrics["mrr"] = 1.0 / (i + 1)
                break
        else:
            metrics["mrr"] = 0

        return metrics

    def evaluate_generation(self, generated_answer: str, reference_answer: str):
        rouge_scores = self.rouge_scorer.score(generated_answer, reference_answer)
        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "em": self.evaluate_em(generated_answer, reference_answer),
            "substring_match": self.evaluate_substring_match(generated_answer, reference_answer)
        }

    def evaluate_em(self, generated_answer: str, reference_answer: str):
        return 1.0 if generated_answer == reference_answer else 0.0

    def evaluate_substring_match(self, generated_answer: str, reference_answer: str):
        return 1.0 if generated_answer.lower() in reference_answer.lower() else 0.0


class RAGEvaluationResults:
    def __init__(self, individual_results: Dict[str, Any], average_metrics: Dict[str, Any]):
        self.individual_results = individual_results
        self.average_metrics = average_metrics

    @classmethod
    def from_dict(cls, results_dict: Dict[str, Any]):
        return cls(
            individual_results=results_dict['individual_results'],
            average_metrics=results_dict['average_metrics']
        )

    def to_dict(self):
        return {
            "individual_results": self.individual_results,
            "average_metrics": self.average_metrics
        }

    def to_table(self):
        retrieval_metrics = self.average_metrics['retrieval']
        retrieval_table = tabulate([
            ["Hit Rate", retrieval_metrics['hit_rate']],
            ["MRR", retrieval_metrics['mrr']]
        ], headers=["Metric", "Value"], 
        tablefmt="grid",
        floatfmt=".3f")

        generation_metrics = self.average_metrics['generation']
        generation_table = tabulate([
            ["ROUGE-1", generation_metrics['rouge1']],
            ["ROUGE-L", generation_metrics['rougeL']]
        ], headers=["Metric", "Value"], 
        tablefmt="grid",
        floatfmt=".3f")

        log("Retrieval Metrics:")
        log(retrieval_table)
        log("\nGeneration Metrics:")
        log(generation_table)
        

def evaluate_rag_results(results, dataset):
    evaluation_results = {}
    evaluator = RAGEvaluator()

    for i, result in results.items():
        reference_answer = dataset["train"][int(i)]["answer"]

        retrieval_metrics = evaluator.evaluate_retrieval(
            retrieved_doc_ids=result["found_ids"], 
            relevant_doc_id=int(i)
        )

        generation_metrics = evaluator.evaluate_generation(
            generated_answer=result["model_answer"], 
            reference_answer=reference_answer
        )

        evaluation_results[i] = {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
        }

    avg_metrics = {"retrieval": {}, "generation": {}}

    for metric in ["hit_rate", "mrr"]:
        avg_metrics["retrieval"][metric] = np.mean(
            [res["retrieval"][metric] for res in evaluation_results.values()]
        )

    for metric in ["rouge1", "rougeL", "em", "substring_match"]:
        avg_metrics["generation"][metric] = np.mean(
            [res["generation"][metric] for res in evaluation_results.values()]
        )

    return RAGEvaluationResults(evaluation_results, avg_metrics)