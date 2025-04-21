import nltk

from .semantic_graph import SemanticGraph

nltk.download("averaged_perceptron_tagger_eng")

__all__ = [
    "SemanticGraph",
]
