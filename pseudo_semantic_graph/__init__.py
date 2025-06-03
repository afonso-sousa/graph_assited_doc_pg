import nltk

from .dependency_tree import TreeNode
from .semantic_graph import SemanticGraph

nltk.download("averaged_perceptron_tagger_eng")

__all__ = [
    "SemanticGraph",
    "TreeNode",
]
