import json
from typing import Dict, List, Tuple, Union

from spacy.tokens import Span, Token

from ..annotations import ARGUMENT_LABELS, CONJ, MODIFIER_LABELS
from ..visualization_mixin import VisualizationMixin
from .prune_and_merge import PruneAndMergeMixin
from .rearrange import RearrangeMixin


class TreeNode(PruneAndMergeMixin, RearrangeMixin, VisualizationMixin):
    def __init__(
        self,
        *,
        word: List[str],
        index: List[int],
        onto_tag: str,
        dep: str,
    ):
        self.word = word
        self.index = index
        self.onto_tag = onto_tag  # Semantic category
        self.dep = dep            # Dependency label

        self.arguments: List["TreeNode"] = []
        self.modifiers: List["TreeNode"] = []

        self.semantic_role: str = ""
        self.predicate: str = ""
        self.edge_label_to_parent: str = dep

    def add_argument(self, node: "TreeNode") -> None:
        self.arguments.append(node)

    def add_arguments(self, nodes: List["TreeNode"]) -> None:
        self.arguments.extend(nodes)

    def add_modifier(self, node: "TreeNode") -> None:
        self.modifiers.append(node)

    def add_modifiers(self, nodes: List["TreeNode"]) -> None:
        self.modifiers.extend(nodes)

    @property
    def children(self) -> List["TreeNode"]:
        return self.arguments + self.modifiers

    def remove_child(self, child: "TreeNode") -> None:
        if child in self.arguments:
            self.arguments.remove(child)
        elif child in self.modifiers:
            self.modifiers.remove(child)

    def __str__(self):
        return json.dumps(self._print_tree(), indent=2)

    def _print_tree(self, indent=0):
        node_dict = {
            "word": self.word,
            "index": self.index,
            "onto_tag": self.onto_tag,
            "dep": self.dep,
            "arguments": [],
            "modifiers": [],
        }

        for child in self.arguments:
            node_dict["arguments"].append(child._print_tree(indent + 2))
        for child in self.modifiers:
            node_dict["modifiers"].append(child._print_tree(indent + 2))

        return node_dict

    @classmethod
    def find_root(cls, sentence: Span) -> Union[None, Token]:
        for token in sentence:
            if token.dep_ == "ROOT":
                return token
        return None

    @classmethod
    def from_spacy(cls, sentence: Span) -> "TreeNode":
        def recursive_build_tree(token):
            breakpoint()
            
            node = cls(
                word=[token.text],
                index=[token.i],
                onto_tag=token.tag_,
                dep=token.dep_,
            )
            for child in token.children:
                if child.dep_ in ARGUMENT_LABELS or (
                    token.dep_ in ARGUMENT_LABELS and child.dep_ in CONJ
                ):
                    node.add_argument(recursive_build_tree(child))
                elif child.dep_ in MODIFIER_LABELS or child.dep_ in CONJ:
                    node.add_modifier(recursive_build_tree(child))
                else:
                    # fallback: treat unknowns as modifiers
                    node.add_modifier(recursive_build_tree(child))
            return node

        root_token = cls.find_root(sentence)
        if root_token is None:
            return None
        return recursive_build_tree(root_token)

    def generate_graph(
        self,
    ) -> Tuple[List[Dict], List[Tuple[int, int, str]]]:
        nodes = []
        edges = []

        def traverse(node, parent_index):
            nonlocal nodes, edges

            # Create a dictionary for the current node
            node_dict = {
                "word": node.word,
                "index": node.index,
                "onto_tag": node.onto_tag,
                "dep": node.dep,
                "semantic_role": node.semantic_role,
                "predicate": node.predicate,
            }
            # Append the current node dictionary to the list of nodes
            nodes.append(node_dict)

            for child in node.children:
                child_index = len(nodes)
                # Append an edge tuple to the list of edges
                edges.append((parent_index, child_index, child.dep))
                # Recursively traverse the child node
                traverse(child, child_index)

        traverse(self, 0)
        return nodes, edges
