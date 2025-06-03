from typing import Dict, List, Optional, Tuple, Union

import spacy
from nltk.corpus import stopwords
from spacy.tokens import Doc

from .annotations import (ARGUMENT_LABELS, CONTENT_POS, MODIFIER_LABELS,
                          MONTHS, RELATIONS_MAPPING)
from .dependency_tree import TreeNode
from .linearize import LinearizationMixin
from .visualization_mixin import VisualizationMixin


class SemanticNode:
    def __init__(
        self, 
        *, 
        word: List[str], 
        index: List[int], 
        onto_tag: str,
        dep: str,
        semantic_role: str = "",
        predicate: str = ""
    ):
        self.word = word
        self.index = index
        self.onto_tag = onto_tag
        self.dep = dep
        self.semantic_role = semantic_role
        self.predicate = predicate

    def __str__(self):
        return f"SemanticNode({self.word}, roles={self.semantic_role}, pred={self.predicate})"


class SemanticGraph(LinearizationMixin, VisualizationMixin):
    nlp = None

    def __init__(
        self,
        nodes: List[Union[SemanticNode, Dict]] = [],
        edges: List[Union[Tuple, Dict]] = [],
        coreferences: Optional[Dict] = None
    ):
        self.nodes = [
            node if isinstance(node, SemanticNode) else SemanticNode(**node)
            for node in nodes
        ]
        self.edges = [
            (edge["from"], edge["to"], edge["relation"]) if isinstance(edge, dict) else edge
            for edge in edges
        ]
        self.coreferences = coreferences or {}

    def __len__(self):
        return len(self.nodes)

    @staticmethod
    def resolve_coreferences(text: Doc) -> str:
        new_text: List[str] = []
        for token in text:
            coref_value = text._.coref_chains.resolve(token)
            if coref_value and len(coref_value) == 1 and token.text.lower() not in ["they", "them", "their"]: # corref does not work well with plural pronouns
                new_text.append(coref_value[0].text)
            else:
                new_text.append(token.text)

        original_tokens = [token.text for token in text]
        breakpoint()
        assert len(new_text) == len(original_tokens), f"Length mismatch: {len(new_text)} != {len(original_tokens)}"
        return " ".join(new_text)

    @classmethod
    def from_text(cls, text):
        if not cls.nlp:
            print("Loading spacy model...")
            cls.nlp = spacy.load("en_core_web_trf")
            cls.nlp.add_pipe("coreferee")

        doc = cls.nlp(text)
        coref_text = cls.resolve_coreferences(doc)
        doc = cls.nlp(coref_text)

        graphs = []
        for sentence in doc.sents:
            tree = TreeNode.from_spacy(sentence)
            if tree is None:
                return None
            tree.prune_and_merge()
            tree.rearrange()
            graph = cls(*tree.generate_graph())
            nodes, edges = graph.nodes, graph.edges
            
            graph = cls(nodes, edges)
            graphs.append(graph)

        return coref_text, cls.merge_graphs(graphs)

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, start_node, end_node, relation):
        if start_node >= len(self.nodes) or end_node >= len(self.nodes):
            raise ValueError("Both nodes must be in the graph.")
        self.edges.append((start_node, end_node, relation))

    def get_children(self, node_idx):
        return [(end_idx, self.nodes[end_idx], rel) for s_idx, end_idx, rel in self.edges if s_idx == node_idx]

    def __str__(self):
        nodes_str = "".join(f"{n.word}, {n.index}, {n.onto_tag}, {n.dep}\n" for n in self.nodes)
        edges_str = "".join(f"{self.nodes[a]} -{r}-> {self.nodes[b]}\n" for a, b, r in self.edges)
        return f"Nodes:\n{nodes_str}\nEdges:\n{edges_str}"

    @classmethod
    def find_similar(cls, nodes, edges):
        stop_words = stopwords.words("english")
        
        # Map: key -> representative node index
        word_to_index = {}

        def is_entity_like(pos_tag: str) -> bool:
            return pos_tag.startswith("NN") or pos_tag.startswith("PRP")

        def normalize(words: List[str], pos_tag: str) -> str:
            # For nouns/pronouns, keep all tokens; otherwise strip stopwords
            if is_entity_like(pos_tag):
                filtered = words
            else:
                filtered = [w for w in words if w.lower() not in stop_words]
            return " ".join(sorted(w.lower() for w in filtered))

        def is_merge_candidate(i: int, j: int) -> bool:
            node_i = nodes[i]
            node_j = nodes[j]

            if not node_i.word or not node_j.word:
                return False

            norm_i = normalize(node_i.word, node_i.onto_tag)
            norm_j = normalize(node_j.word, node_j.onto_tag)
            common = set(norm_i.split()) & set(norm_j.split())
            if not common:
                return False

            # Must be content-bearing nodes
            pos_ok = node_i.onto_tag in CONTENT_POS and node_j.onto_tag in CONTENT_POS
            dep_ok = any(rel in ARGUMENT_LABELS + MODIFIER_LABELS for rel in [node_i.dep, node_j.dep])
            case_ok = any(w[0].isupper() for w in node_i.word + node_j.word)

            ratio = len(common) / max(1, min(len(norm_i.split()), len(norm_j.split())))
            return (pos_ok or dep_ok) and (case_ok or ratio > 0.5)

        for i in range(len(nodes)):
            norm_i = normalize(nodes[i].word, nodes[i].onto_tag)
            if not norm_i:
                continue

            breakpoint()
            for j in range(i + 1, len(nodes)):
                norm_j = normalize(nodes[j].word, nodes[j].onto_tag)
                if norm_i == norm_j or is_merge_candidate(i, j):
                    target_idx = word_to_index.get(norm_i, i)
                    word_to_index[norm_j] = target_idx
                    word_to_index[norm_i] = target_idx
                    edges = cls.redirect_from_to(edges, j, target_idx)

        return edges

    @staticmethod
    def redirect_from_to(edges, from_idx, to_idx):
        return [
            (to_idx if s == from_idx else s, to_idx if t == from_idx else t, r)
            for s, t, r in edges
        ]

    @classmethod
    def simplify_relations(cls, edges):
        return [(s, t, RELATIONS_MAPPING.get(r, r)) for s, t, r in edges]

    @classmethod
    def set_date_relations(cls, nodes, edges):
        date_nodes = [i for i, n in enumerate(nodes) if any(w.lower() in MONTHS for w in n.word)]
        return [(s, t, "date") if t in date_nodes else (s, t, r) for s, t, r in edges]

    @classmethod
    def remove_dangling_nodes(cls, nodes, edges):
        if len(nodes) == 1:
            return nodes, edges
        used_nodes = set([s for s, _, _ in edges] + [t for _, t, _ in edges])
        nodes_with_edges = [node for idx, node in enumerate(nodes) if idx in used_nodes]
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate([i for i in range(len(nodes)) if i in used_nodes])}
        new_edges = [(node_mapping[s], node_mapping[t], r) for s, t, r in edges if s in node_mapping and t in node_mapping]
        return nodes_with_edges, new_edges

    @classmethod
    def merge_graphs(cls, graphs):
        total_nodes, total_edges = [], []
        offset = 0
        for graph in graphs:
            total_nodes.extend(graph.nodes)
            total_edges.extend([(s + offset, t + offset, r) for s, t, r in graph.edges])
            offset += len(graph.nodes)

        # total_edges = cls.find_similar(total_nodes, total_edges)
        # total_nodes, total_edges = cls.remove_dangling_nodes(total_nodes, total_edges)
        # total_edges = cls.simplify_relations(total_edges)
        # total_edges = cls.set_date_relations(total_nodes, total_edges)
        return cls(total_nodes, total_edges)
