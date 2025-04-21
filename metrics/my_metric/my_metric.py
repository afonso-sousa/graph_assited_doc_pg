import statistics

import datasets
import evaluate
from parascore import ParaScorer
from tqdm.auto import tqdm

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MyMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "sources": datasets.Value("string"),
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
        )

    def _download_and_prepare(self, dl_manager):
        self.bleu = evaluate.load("bleu", experiment_id=self.experiment_id)
        self.rouge = evaluate.load("rouge", experiment_id=self.experiment_id)
        self.sbert = evaluate.load("metrics/sbert", experiment_id=self.experiment_id)
        self.para_score = ParaScorer(
            lang="en", model_type="bert-base-cased-finetuned-mrpc"
        )

    def _compute(self, sources, predictions, references, compute_pair_wise=False):
        if not compute_pair_wise:
            bleu_score = self.bleu.compute(
                predictions=predictions,
                references=[[ref] for ref in references],
            )["bleu"]

            self_bleu_score = self.bleu.compute(
                predictions=predictions, references=[[src] for src in sources]
            )["bleu"]

            alpha = 0.7
            ibleu_formula = lambda bleu, self_bleu: round(
                alpha * bleu - (1 - alpha) * self_bleu, 3
            )
            ibleu_score = ibleu_formula(bleu_score, self_bleu_score)

            rouge_score = self.rouge.compute(
                predictions=predictions,
                references=references,
            )

            sbert_scores = self.sbert.compute(
                predictions=predictions, references=references
            )["scores"]

            para_scores = self.para_score.base_score(
                predictions, sources, references, batch_size=16
            )

            return {
                "bleu": bleu_score * 100,
                "self_bleu": self_bleu_score * 100,
                "ibleu": ibleu_score * 100,
                "rouge1": round(rouge_score["rouge1"] * 100, 3),
                "rouge2": round(rouge_score["rouge2"] * 100, 3),
                "rougeL": round(rouge_score["rougeL"] * 100, 3),
                "sbert_mean": statistics.mean(sbert_scores) * 100,
                "sbert_std": (
                    statistics.stdev(sbert_scores) * 100 if len(sbert_scores) > 1 else 0
                ),
                "para_score_mean": statistics.mean(para_scores) * 100,
                "para_score_std": (
                    statistics.stdev(para_scores) * 100 if len(para_scores) > 1 else 0
                ),
            }
        else:
            total_len = len(predictions)
            bleu_scores = [
                self.bleu.compute(predictions=[pred], references=[[ref]])["bleu"]
                for pred, ref in tqdm(
                    zip(predictions, references), total=total_len, desc="bleu"
                )
            ]

            self_bleu_scores = [
                self.bleu.compute(predictions=[pred], references=[[src]])["bleu"]
                for pred, src in tqdm(
                    zip(predictions, sources),
                    total=total_len,
                    desc="self_bleu",
                )
            ]
            alpha = 0.7
            ibleu_formula = lambda scores: round(
                alpha * scores[0] - (1 - alpha) * scores[1], 3
            )
            ibleu_scores = list(map(ibleu_formula, zip(bleu_scores, self_bleu_scores)))

            rouge_scores = [
                self.rouge.compute(predictions=[pred], references=[ref])
                for pred, ref in tqdm(
                    zip(predictions, references), total=total_len, desc="rouge"
                )
            ]

            sbert_scores = self.sbert.compute(
                predictions=predictions, references=references
            )["scores"]

            para_scores = self.para_score.base_score(
                predictions, sources, references, batch_size=16
            )

            return {
                "bleu": bleu_scores,
                "self_bleu": self_bleu_scores,
                "ibleu": ibleu_scores,
                "rouge1": [entry["rouge1"] for entry in rouge_scores],
                "rouge2": [entry["rouge2"] for entry in rouge_scores],
                "rougeL": [entry["rougeL"] for entry in rouge_scores],
                "sbert": sbert_scores,
                "para_score": para_scores,
            }
