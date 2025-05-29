from typing import Dict, List, Tuple, Union

import nltk
import spacy
from nltk.corpus import stopwords
from spacy.tokens import Doc, Span

from .annotations import *
from .dependency_tree import TreeNode
from .linearize import LinearizationMixin


class Node:
    def __init__(
        self, *, word: List[str], index: List[int], ud_pos: str, onto_tag: str
    ):
        self.word = word
        self.index = index
        self.ud_pos = ud_pos
        self.onto_tag = onto_tag

    def __str__(self):
        return f"Node({self.word}, {self.index}, {self.ud_pos}, {self.onto_tag})"


class SemanticGraph(LinearizationMixin):
    nlp = None

    def __init__(
        self,
        nodes: List[Union[Node, Dict]] = [],
        edges: List[Union[Tuple, Dict]] = [],
    ):
        if not nodes:
            self.nodes = []
            self.edges = []
        else:
            if isinstance(nodes[0], dict):
                self.nodes = [Node(**d) for d in nodes]
            else:
                self.nodes = nodes

            if edges and isinstance(edges[0], dict):
                self.edges = [(e["start"], e["end"], e["relation"]) for e in edges]
            else:
                self.edges = edges

    @staticmethod
    def resolve_with_coref(text: Doc) -> str:
        new_text: List[str] = []
        for token in text:
            coref_value = text._.coref_chains.resolve(token)
            if coref_value:
                if len(coref_value) > 1:
                    new_text.append(" and ".join([i.text for i in coref_value]))
                else:
                    new_text.append(coref_value[0].text)
            else:
                new_text.append(token.text)
        return " ".join(new_text)

    @classmethod
    def from_text(cls, text):
        if not cls.nlp:
            print("Loading spacy model...")
            cls.nlp = spacy.load("en_core_web_trf")
            cls.nlp.add_pipe("coreferee")

        coref_text = cls.resolve_with_coref(cls.nlp(text))

        sentences: List[Span] = cls.nlp(coref_text).sents
        graphs = []
        for sentence in sentences:
            tree = TreeNode.from_spacy(sentence)
            if tree is None:
                return None
            print(tree)
            tree.prune_and_merge()
            print("-" * 20)
            print(tree)
            tree.rearrange()
            graph = cls(*tree.generate_graph())
            graphs.append(graph)

        return cls.merge_graphs(graphs)

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, start_node, end_node, relation):
        if start_node >= len(self.nodes) or end_node >= len(self.nodes):
            raise ValueError("Both nodes must be in the graph.")
        self.edges.append((start_node, end_node, relation))

    def get_children(self, node_idx):
        children = []
        for edge in self.edges:
            start_idx, end_idx, relation = edge
            if start_idx == node_idx:
                children.append((end_idx, self.nodes[end_idx], relation))
        return children

    def __str__(self):
        nodes_str = ""
        for node in self.nodes:
            nodes_str += f"{node.word}, {node.index}, {node.ud_pos}, {node.onto_tag}\n"

        edges_str = ""
        for edge in self.edges:
            edges_str += f"{self.nodes[edge[0]]} -{edge[2]}-> {self.nodes[edge[1]]}\n"

        return f"Nodes:\n{nodes_str}\nEdges:\n{edges_str}"

    def visualize(self):
        from graphviz import Graph

        graph_viz = Graph()
        for i, node in enumerate(self.nodes):
            shape = "ellipse"
            if node.onto_tag in VERB_POS:
                shape = "diamond"
            elif node.ud_pos == "PROPN":
                shape = "box"
            elif node.ud_pos == "NOUN":
                shape = "parallelogram"

            graph_viz.node(f"{i}", " ".join(node.word), shape=shape)

        for edge in self.edges:
            graph_viz.edge(f"{edge[0]}", f"{edge[1]}", label=edge[2])

        return graph_viz

    @classmethod
    def find_similar(cls, nodes, edges):
        # new_edges = []
        stopwords_eng = stopwords.words("english")
        for i in range(len(nodes)):
            word_i = [w for w in nodes[i].word if w not in set(stopwords_eng)]
            for j in range(i + 1, len(nodes)):
                word_j = [w for w in nodes[j].word if w not in set(stopwords_eng)]
                common = list(set(word_i) & set(word_j))
                if not common:
                    continue

                pos_list = list(zip(*nltk.pos_tag(common)))[1]

                flag_pos = any(pos in NOUN_POS + MODIFIER_POS for pos in pos_list)

                flag_up = any(not w.islower() for w in common)

                pos_qualify = (
                    nodes[i].onto_tag in PREP_POS + VERB_POS + NOUN_POS
                    and nodes[j].onto_tag in PREP_POS + VERB_POS + NOUN_POS
                )

                has_dep_i = any(
                    t[2]
                    in SUBJ_AND_OBJ
                    + [
                        "amod",
                        "nn",
                        "mwe",
                    ]
                    for t in edges
                    if t[1] == i
                )

                has_dep_j = any(
                    t[2]
                    in SUBJ_AND_OBJ
                    + [
                        "amod",
                        "nn",
                        "mwe",
                    ]
                    for t in edges
                    if t[1] == j
                )

                dep_qualify = has_dep_i and has_dep_j

                if pos_qualify or dep_qualify:
                    if (flag_up or flag_pos) and len(word_i) * len(word_j) > 0:
                        prb1, prb2 = len(word_i) / len(common), len(word_j) / len(
                            common
                        )
                        if max(prb1, prb2) > 1 / 2 and min(prb1, prb2) > 1 / 3:
                            # new_edges.append((i, j, "SIMILAR"))
                            edges = cls.redirect_from_to(edges, j, i)

        return edges

    def redirect_from_to(edges, from_idx, to_idx):
        redirected_edges = []

        for source, target, relation in edges:
            # Check if the edge points to 'j'
            if target == from_idx:
                # Redirect the edge to point to 'i' instead of 'j'
                redirected_edges.append((source, to_idx, relation))
            # Check if the edge originates from 'from_idx'
            elif source == from_idx:
                # Redirect the edge to originate from 'to_idx' instead of 'from_idx'
                redirected_edges.append((to_idx, target, relation))
            else:
                # Keep the original edge
                redirected_edges.append((source, target, relation))
        return redirected_edges

    @classmethod
    def simplify_relations(cls, edges):
        new_edges = []
        for edge in edges:
            new_edges.append(
                (edge[0], edge[1], RELATIONS_MAPPING.get(edge[2], edge[2]))
            )

        return new_edges

    @classmethod
    def set_date_relations(cls, nodes, edges):
        date_nodes = []
        for i, node in enumerate(nodes):
            if any(word.lower() in MONTHS for word in node.word):
                date_nodes.append(i)
        new_edges = []
        for edge in edges:
            if edge[1] in date_nodes:
                new_edges.append((edge[0], edge[1], "date"))
            else:
                new_edges.append(edge)

        return new_edges

    @classmethod
    def convert_verbs_to_base_form(cls, nodes):
        if not cls.nlp:
            print("Loading spacy model...")
            cls.nlp = spacy.load("en_core_web_trf")
            cls.nlp.add_pipe("coreferee")
        new_nodes = []
        for node in nodes:
            if node.ud_pos != "VERB":
                new_nodes.append(node)
                continue
            doc = cls.nlp(" ".join(node.word))
            base_form = []
            for token in doc:
                if token.dep_ in ["aux", "auxpass"]:
                    continue
                base_form.append(token.lemma_)
            node.word = base_form
            new_nodes.append(node)

        return new_nodes

    @classmethod
    def remove_dangling_nodes(cls, nodes, edges):
        if len(nodes) == 1:
            return nodes, edges

        used_nodes = set()

        for source, target, _ in edges:
            used_nodes.add(nodes[source])
            used_nodes.add(nodes[target])

        nodes_with_edges = []
        node_mapping = {}

        for idx, node in enumerate(nodes):
            if node in used_nodes:
                nodes_with_edges.append(node)
                node_mapping[idx] = len(nodes_with_edges) - 1

        # Step 4: Update the edge indexes to match the new node list
        new_edges = [
            (node_mapping[source], node_mapping[target], relation)
            for source, target, relation in edges
        ]

        return nodes_with_edges, new_edges

    @classmethod
    def merge_graphs(cls, graphs):
        total_nodes = []
        total_edges = []
        for graph in graphs:
            max_node_index = len(total_nodes)
            total_nodes.extend(graph.nodes)
            reindexed_edges = [
                (edge[0] + max_node_index, edge[1] + max_node_index, edge[2])
                for edge in graph.edges
            ]
            total_edges.extend(reindexed_edges)

        total_edges = cls.find_similar(total_nodes, total_edges)
        # total_edges.extend(new_edges)
        # total_nodes = cls.convert_verbs_to_base_form(total_nodes)
        total_nodes, total_edges = cls.remove_dangling_nodes(total_nodes, total_edges)
        total_edges = cls.simplify_relations(total_edges)
        total_edges = cls.set_date_relations(total_nodes, total_edges)
        return cls(total_nodes, total_edges)
