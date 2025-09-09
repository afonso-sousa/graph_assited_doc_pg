# %%
from pseudo_semantic_graph import SemanticGraph

s = "He worked in France and Italy."
_, graph = SemanticGraph.from_text(s)
graph.visualize()
assert graph.edges == [(0, 1, 'ARG0'), (0, 2, 'prep'), (2, 3, 'ARG1'), (2, 4, 'ARG1')]
assert [node.word for node in graph.nodes] == [['worked'], ['He'], ['in'], ['France'], ['Italy']]

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He worked quickly and efficiently."
_, graph = SemanticGraph.from_text(s)
graph.visualize()
assert graph.edges == [(0, 1, 'ARG0'), (0, 2, 'ARGM'), (0, 3, 'ARGM')]
assert [node.word for node in graph.nodes] == [['worked'], ['He'], ['quickly'], ['efficiently']]

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He worked in France, Italy and Germany."
_, graph = SemanticGraph.from_text(s)
graph.visualize()
assert graph.edges == [(0, 1, 'ARG0'),
 (0, 2, 'prep'),
 (2, 3, 'ARG1'),
 (2, 4, 'ARG1'),
 (2, 5, 'ARG1')]
assert [node.word for node in graph.nodes] == [['worked'], ['He'], ['in'], ['France'], ['Italy'], ['Germany']]

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He worked in France and lived in Italy."
_, graph = SemanticGraph.from_text(s)
graph.visualize()
assert graph.edges == [(0, 1, 'ARG0'),
 (1, 2, 'ARG0'),
 (2, 3, 'prep'),
 (3, 4, 'ARG1'),
 (0, 5, 'prep'),
 (5, 6, 'ARG1')]
assert [node.word for node in graph.nodes] == [['worked'], ['He'], ['lived'], ['in'], ['Italy'], ['in'], ['France']]

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He was born in France."
_, graph = SemanticGraph.from_text(s)
graph.visualize()
assert graph.edges == [(0, 1, 'ARG0'), (0, 2, 'auxpass'), (0, 3, 'prep'), (3, 4, 'ARG1')]
assert [node.word for node in graph.nodes] == [['born'], ['He'], ['was'], ['in'], ['France']]

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He could not help noticing how well she knew the way, and when he suggested making a slight detour because it was less crowded along there, she heard him out as though under strain"
_, graph = SemanticGraph.from_text(s)
graph.visualize()
assert graph.edges == [(0, 1, 'ARG0'),
 (0, 2, 'ARG2'),
 (2, 3, 'ARG2'),
 (3, 4, 'ARG0'),
 (3, 5, 'ARG1'),
 (3, 6, 'ARGM'),
 (6, 7, 'ARGM'),
 (0, 8, 'neg'),
 (9, 4, 'ARG0'),
 (9, 1, 'ARG1'),
 (9, 10, 'ARGM'),
 (10, 1, 'ARG0'),
 (10, 11, 'ARG2'),
 (11, 12, 'ARG1'),
 (11, 13, 'ARGM'),
 (13, 12, 'ARG0'),
 (13, 14, 'ARG2'),
 (11, 15, 'prep'),
 (15, 16, 'pcomp'),
 (10, 17, 'ARGM'),
 (9, 18, 'prt'),
 (9, 19, 'prep'),
 (19, 20, 'pcomp'),
 (19, 21, 'prep'),
 (21, 22, 'ARG1')]
assert [node.word for node in graph.nodes] == [['help'],
 ['He'],
 ['noticing'],
 ['knew'],
 ['she'],
 ['way'],
 ['well'],
 ['how'],
 ['not'],
 ['heard'],
 ['suggested'],
 ['making'],
 ['slight', 'detour'],
 ['was'],
 ['less', 'crowded'],
 ['along'],
 ['there'],
 ['when'],
 ['out'],
 ['as'],
 ['though'],
 ['under'],
 ['strain']]


# # %%
# from pseudo_semantic_graph import SemanticGraph

# t = 'He stopped dead immediately, unable to say anything more. That was his one and only attempt to bring Aglaya to her senses; he then followed her slavishly. However disordered his thoughts were, he realized that nothing would stop her from going there, and consequently he was duty-bound to go with her. He could tell the strength of her resolve and how futile it would be for him to stand in her way. They walked in silence, hardly a word passed between them. He could not help noticing how well she knew the way, and when he suggested making a slight detour because it was less crowded along there, she heard him out as though under strain, and replied curtly, "Makes no difference!" As they drew level with Darya Alexeyevna\'s large old timber house, a splendidly attired lady in the company of a much younger woman emerged from the front door. They both got into a magnificent calash, which was waiting in the driveway, talking and laughing loudly, without so much as a glance at the two approaching figures as though they had not even noticed them. As soon as the calash had moved off, the door opened for a second time, and Rogozhin, who was already expecting them, let them in and shut it behind them.'

# coref_text, g = SemanticGraph.from_text(
#     t
# )
# print(g)

# %%
