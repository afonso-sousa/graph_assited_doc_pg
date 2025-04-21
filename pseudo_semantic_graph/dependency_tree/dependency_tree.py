import json
from typing import Dict, List, Tuple, Union

from spacy.tokens import Span, Token

from ..annotations import *
from .prune_and_merge import PruneAndMergeMixin
from .rearrange import RearrangeMixin


class TreeNode(PruneAndMergeMixin, RearrangeMixin):
    def __init__(
        self,
        *,
        word: List[str],
        index: List[int],
        ud_pos: str,
        onto_tag: str,
        dep: str,
    ):
        self.word = word
        self.index = index
        self.ud_pos = ud_pos
        self.onto_tag = onto_tag
        self.dep = dep
        self.nouns: List["TreeNode"] = []
        self.verbs: List["TreeNode"] = []
        self.attributes: List["TreeNode"] = []

    def add_noun(self, noun: "TreeNode") -> None:
        self.nouns.append(noun)

    def add_nouns(self, nouns: List["TreeNode"]) -> None:
        self.nouns.extend(nouns)

    def add_verb(self, verb: "TreeNode") -> None:
        self.verbs.append(verb)

    def add_verbs(self, verbs: List["TreeNode"]) -> None:
        self.verbs.extend(verbs)

    def add_attribute(self, attribute: "TreeNode") -> None:
        self.attributes.append(attribute)

    def add_attributes(self, attributes: List["TreeNode"]) -> None:
        self.attributes.extend(attributes)

    @property
    def children(self) -> List["TreeNode"]:
        return self.nouns + self.verbs + self.attributes

    def remove_child(self, child: "TreeNode") -> None:
        if child in self.nouns:
            self.nouns.remove(child)
        elif child in self.verbs:
            self.verbs.remove(child)
        elif child in self.attributes:
            self.attributes.remove(child)

    def __str__(self):
        return json.dumps(self._print_tree(), indent=2)

    def _print_tree(self, indent=0):
        node_dict = {
            "word": self.word,
            "index": self.index,
            "ud_pos": self.ud_pos,
            "onto_tag": self.onto_tag,
            "dep": self.dep,
            "nouns": [],
            "verbs": [],
            "attributes": [],
        }

        for child in self.nouns:
            node_dict["nouns"].append(child._print_tree(indent + 2))
        for child in self.verbs:
            node_dict["verbs"].append(child._print_tree(indent + 2))
        for child in self.attributes:
            node_dict["attributes"].append(child._print_tree(indent + 2))

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
            def is_noun(child: Token) -> bool:
                return child.dep_ in SUBJ_AND_OBJ or (
                    token.dep_ in SUBJ_AND_OBJ and child.dep_ in CONJ
                )

            def is_verb(child: Token) -> bool:
                return child.pos_ in ["VERB", "AUX"]

            head_index_map = {token.i: idx for idx, token in enumerate(sentence)}
            node = cls(
                word=[token.text],
                index=[head_index_map[token.i]],
                ud_pos=token.pos_,
                onto_tag=token.tag_,
                dep=token.dep_,
            )
            for child in token.children:
                if is_noun(child):
                    node.add_noun(recursive_build_tree(child))
                elif is_verb(child):
                    node.add_verb(recursive_build_tree(child))
                else:
                    node.add_attribute(recursive_build_tree(child))
            return node

        root_token = cls.find_root(sentence)
        if root_token is None:
            return None
        return recursive_build_tree(root_token)

    def rearrange(self) -> None:
        # rearrange nouns
        self.redirect_nominal_conjuncts()
        self.redirect_attribute_conjuncts()

        # rearrange verbs
        self.redirect_verbal_conjuncts()
        self.redirect_root_conj_children()

        # rearrange attributes
        self.merge_prepositions()

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
                "ud_pos": node.ud_pos,
                "onto_tag": node.onto_tag,
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
