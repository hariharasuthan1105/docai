"""
Document AI Evaluation, Benchmarking, and Ablation Framework.
"""

from docai.evaluation.dataset import create_sample_evaluation_dataset, load_evaluation_dataset
from docai.evaluation.evaluator import DocumentAIEvaluator, EvaluationMetrics
from docai.evaluation.ablation import run_ablation_study
from docai.evaluation.robustness import run_robustness_benchmark

__all__ = [
    "create_sample_evaluation_dataset",
    "load_evaluation_dataset",
    "DocumentAIEvaluator",
    "EvaluationMetrics",
    "run_ablation_study",
    "run_robustness_benchmark",
]
