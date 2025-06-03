from graphviz import Graph

from .annotations import NOUN_TAGS, PROPN_TAGS, VERB_TAGS


class VisualizationMixin:

    def visualize(self):
        graph_viz = Graph()
        if hasattr(self, 'generate_graph'):
            nodes, edges = self.generate_graph()
        else:
            nodes, edges = self.nodes, self.edges
        
        for i, node in enumerate(nodes):
            shape = "ellipse"
            onto_tag = node.onto_tag if hasattr(node, 'onto_tag') else node['onto_tag']
            word = node.word if hasattr(node, 'word') else node['word']

            if onto_tag in VERB_TAGS:
                shape = "diamond"
            elif onto_tag in PROPN_TAGS:
                shape = "box"
            elif onto_tag in NOUN_TAGS:
                shape = "parallelogram"

            graph_viz.node(f"{i}", " ".join(word), shape=shape)

        for edge in edges:
            graph_viz.edge(f"{edge[0]}", f"{edge[1]}", label=edge[2])

        return graph_viz