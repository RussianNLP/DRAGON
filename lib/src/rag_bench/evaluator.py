import ast
import re
from types import SimpleNamespace
from typing import Any, Dict
import warnings

import numpy as np
from rouge_score import rouge_scorer
from tabulate import tabulate
from nltk.stem.snowball import SnowballStemmer


from .helper import log
from .constants import THINK_END_TOKEN


class RAGEvaluator:
    def __init__(self):
        self.rus_stem = SnowballStemmer("russian")

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            tokenizer=SimpleNamespace(tokenize=self.tokenize_ru),
            use_stemmer=False
        )

    def tokenize_ru(self, text: str):
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return [self.rus_stem.stem(t) for t in tokens]

    @staticmethod
    def evaluate_retrieval(retrieved_doc_ids, relevant_doc_ids):
        metrics = dict()

        # hit rate
        metrics["hit_rate"] = len(set(relevant_doc_ids) & set(retrieved_doc_ids)) / len(relevant_doc_ids)

        # mrr
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                metrics["mrr"] = 1.0 / (i + 1)
                break
        else:
            metrics["mrr"] = 0

        return metrics

    def evaluate_generation(self, generated_answer: str, reference_answer: str):
        rouge_scores = self.rouge_scorer.score(generated_answer, reference_answer)
        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "exact_match": self.evaluate_em(
                generated_answer=self.normalize_text(generated_answer),
                reference_answer=self.normalize_text(reference_answer)
            ),
            "substring_match": self.evaluate_substring_match(
                generated_answer=self.normalize_text(generated_answer),
                reference_answer=self.normalize_text(reference_answer)
            )
        }

    @staticmethod
    def normalize_text(s: str):
        return " ".join(s.strip().split())

    @staticmethod
    def evaluate_em(generated_answer: str, reference_answer: str):
        return 1.0 if generated_answer.strip() == reference_answer.strip() else 0.0

    @staticmethod
    def evaluate_substring_match(generated_answer: str, reference_answer: str):
        return 1.0 if generated_answer.lower() in reference_answer.lower() else 0.0


class RAGEvaluationResults:
    def __init__(self, individual_results: Dict[str, Any], average_metrics: Dict[str, Any]):
        self.individual_results = individual_results
        self.average_metrics = average_metrics

    @classmethod
    def from_dict(cls, results_dict: Dict[str, Any]):
        return cls(
            individual_results=results_dict["individual_results"],
            average_metrics=results_dict["average_metrics"]
        )

    def to_dict(self):
        return {
            "individual_results": self.individual_results,
            "average_metrics": self.average_metrics
        }

    def to_table(self, overall_only: bool = True):
        if overall_only:
            retrieval_metrics = self.average_metrics["overall"]["retrieval"]
            retrieval_table = tabulate([
                ["Hit Rate", retrieval_metrics["hit_rate"]],
                ["MRR", retrieval_metrics["mrr"]]
            ], headers=["Metric", "Value"],
                tablefmt="grid",
                floatfmt=".4f")

            generation_metrics = self.average_metrics["overall"]["generation"]
            generation_table = tabulate([
                ["ROUGE-1", generation_metrics["rouge1"]],
                ["ROUGE-2", generation_metrics["rouge2"]],
                ["ROUGE-L", generation_metrics["rougeL"]],
                ["Exact Match", generation_metrics["exact_match"]],
                ["Substring Match", generation_metrics["substring_match"]]
            ], headers=["Metric", "Value"],
                tablefmt="grid",
                floatfmt=".4f")
        else:
            retrieval_metrics = self.average_metrics["overall"]["retrieval"]
            table_content = [
                ["Hit Rate", retrieval_metrics["hit_rate"]],
                ["MRR", retrieval_metrics["mrr"]]
            ]
            table_header = ["Metric", "Overall"]
            for question_type in sorted(list(set(self.average_metrics.keys()) - {"overall"})):
                retrieval_metrics = self.average_metrics[question_type]["retrieval"]
                table_content[0].append(retrieval_metrics["hit_rate"])
                table_content[1].append(retrieval_metrics["mrr"])
                table_header.append(question_type)
            retrieval_table = tabulate(table_content, headers=table_header, tablefmt="grid", floatfmt=".4f")
            del table_content, table_header

            generation_metrics = self.average_metrics["overall"]["generation"]
            table_content = [
                ["ROUGE-1", generation_metrics["rouge1"]],
                ["ROUGE-2", generation_metrics["rouge2"]],
                ["ROUGE-L", generation_metrics["rougeL"]],
                ["Exact Match", generation_metrics["exact_match"]],
                ["Substring Match", generation_metrics["substring_match"]]
            ]
            table_header = ["Metric", "Overall"]
            for question_type in sorted(list(set(self.average_metrics.keys()) - {"overall"})):
                generation_metrics = self.average_metrics[question_type]["generation"]
                table_content[0].append(generation_metrics["rouge1"])
                table_content[1].append(generation_metrics["rouge2"])
                table_content[2].append(generation_metrics["rougeL"])
                table_content[3].append(generation_metrics["exact_match"])
                table_content[4].append(generation_metrics["substring_match"])
                table_header.append(question_type)
            generation_table = tabulate(table_content, headers=table_header, tablefmt="grid", floatfmt=".4f")
            del table_content, table_header

        log("Retrieval Metrics:")
        log(retrieval_table)
        log("\nGeneration Metrics:")
        log(generation_table)
        return "Retrieval Metrics:\n" + retrieval_table + "\n\nGeneration Metrics:\n" + generation_table + "\n"


def check_doc_ids(doc_ids) -> bool:
    if not isinstance(doc_ids, list):
        return False
    ok = True
    for val in doc_ids:
        if not isinstance(val, int):
            ok = False
            break
    return ok


def evaluate_rag_results(results, dataset, text_mapping):
    evaluation_results = dict()
    evaluator = RAGEvaluator()
    question_types = sorted(list(set(dataset["train"]["type"])))

    for test_sample in dataset["train"]:
        question_type = test_sample["type"]
        public_id = test_sample["public_id"]
        if (not isinstance(public_id, str)) and (not isinstance(public_id, int)):
            err_msg = (f"The public_id {public_id} is wrong! Expected {type('123')} or {type(123)}, "
                       f"got {type(public_id)}.")
            raise ValueError(err_msg)
        if isinstance(public_id, int):
            public_id = str(public_id)
        reference_answer = test_sample["answer"]
        if not isinstance(test_sample["text_ids"], str):
            err_msg = (f"The text_ids {test_sample['text_ids']} is wrong! "
                       f"Expected {type('123')}, got {type(test_sample['text_ids'])}.")
            raise ValueError(err_msg)
        text_ids = ast.literal_eval(test_sample["text_ids"])
        set_of_doc_ids = set()
        relevant_doc_ids = []
        for doc_id in text_ids:
            if isinstance(doc_id, list):
                for doc_id__ in doc_id:
                    if isinstance(doc_id__, int):
                        if doc_id__ not in set_of_doc_ids:
                            set_of_doc_ids.add(doc_id__)
                            relevant_doc_ids.append(doc_id__)
                    else:
                        doc_id_ = int(doc_id__)
                        if doc_id_ not in set_of_doc_ids:
                            set_of_doc_ids.add(doc_id_)
                            relevant_doc_ids.append(doc_id_)
            elif isinstance(doc_id, int):
                if doc_id not in set_of_doc_ids:
                    set_of_doc_ids.add(doc_id)
                    relevant_doc_ids.append(doc_id)
            else:
                doc_id_ = int(doc_id)
                if doc_id_ not in set_of_doc_ids:
                    set_of_doc_ids.add(doc_id_)
                    relevant_doc_ids.append(doc_id_)
        del set_of_doc_ids
        if not check_doc_ids(relevant_doc_ids):
            raise RuntimeError(f"The relevant document IDs for question {public_id} are wrong!\n{relevant_doc_ids}")

        try:
            predicted = results[public_id]
        except:
            try:
                predicted = results[int(public_id)]
            except Exception as err:
                warnings.warn(str(err))
                predicted = {
                    "found_ids": [],
                    "model_answer": ""
                }
        if not isinstance(predicted["found_ids"], list):
            err_msg = (f"The found_ids {predicted['text_ids']} is wrong! "
                       f"Expected {type([1, 2, 3])}, got {type(predicted['found_ids'])}.")
            raise ValueError(err_msg)
        predicted_answer = predicted["model_answer"]
        think_end_idx = predicted_answer.find(THINK_END_TOKEN)
        if think_end_idx >= 0:
            predicted_answer = predicted_answer[(think_end_idx + len(THINK_END_TOKEN)):]
        found_doc_ids = [int(text_mapping[public_id_]) for public_id_ in predicted["found_ids"]]
        if not check_doc_ids(found_doc_ids):
            raise RuntimeError(f"The found document IDs for question {public_id} are wrong!\n{found_doc_ids}")

        retrieval_metrics = evaluator.evaluate_retrieval(
            retrieved_doc_ids=found_doc_ids,
            relevant_doc_ids=relevant_doc_ids
        )

        generation_metrics = evaluator.evaluate_generation(
            generated_answer=predicted_answer,
            reference_answer=reference_answer
        )

        evaluation_results[public_id] = {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "question_type": question_type
        }

    avg_metrics = {"overall": {"retrieval": {}, "generation": {}}}
    for metric in ["hit_rate", "mrr"]:
        avg_metrics["overall"]["retrieval"][metric] = np.mean(
            [res["retrieval"][metric] for res in evaluation_results.values()]
        )
    for metric in ["rouge1", "rouge2", "rougeL", "exact_match", "substring_match"]:
        avg_metrics["overall"]["generation"][metric] = np.mean(
            [res["generation"][metric] for res in evaluation_results.values()]
        )

    for question_type in question_types:
        avg_metrics[question_type] = {"retrieval": {}, "generation": {}}
        filtered_values = list(filter(lambda it: it["question_type"] == question_type, evaluation_results.values()))
        for metric in ["hit_rate", "mrr"]:
            avg_metrics[question_type]["retrieval"][metric] = np.mean(
                [res["retrieval"][metric] for res in filtered_values]
            )
        for metric in ["rouge1", "rouge2", "rougeL", "exact_match", "substring_match"]:
            avg_metrics[question_type]["generation"][metric] = np.mean(
                [res["generation"][metric] for res in filtered_values]
            )

    return RAGEvaluationResults(evaluation_results, avg_metrics)
