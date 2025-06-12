# %%
from pseudo_semantic_graph import SemanticGraph

s = "He worked in France and Italy."
_, graph = SemanticGraph.from_text(s)
print(graph)
graph.visualize()

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He worked quickly and efficiently."
_, graph = SemanticGraph.from_text(s)
graph.visualize()

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He worked in France, Italy and Germany."
_, graph = SemanticGraph.from_text(s)
print(graph)
graph.visualize()

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He worked in France and lived in Italy."
_, graph = SemanticGraph.from_text(s)
print(graph)
graph.visualize()

# %%
from pseudo_semantic_graph import SemanticGraph

s = "He was born in France."
_, graph = SemanticGraph.from_text(s)
print(graph)
graph.visualize()

# %%
# from pseudo_semantic_graph import SemanticGraph

# t = 'He stopped dead immediately, unable to say anything more. That was his one and only attempt to bring Aglaya to her senses; he then followed her slavishly. However disordered his thoughts were, he realized that nothing would stop her from going there, and consequently he was duty-bound to go with her. He could tell the strength of her resolve and how futile it would be for him to stand in her way. They walked in silence, hardly a word passed between them. He could not help noticing how well she knew the way, and when he suggested making a slight detour because it was less crowded along there, she heard him out as though under strain, and replied curtly, "Makes no difference!" As they drew level with Darya Alexeyevna\'s large old timber house, a splendidly attired lady in the company of a much younger woman emerged from the front door. They both got into a magnificent calash, which was waiting in the driveway, talking and laughing loudly, without so much as a glance at the two approaching figures as though they had not even noticed them. As soon as the calash had moved off, the door opened for a second time, and Rogozhin, who was already expecting them, let them in and shut it behind them.'

# coref_text, g = SemanticGraph.from_text(
#     t
# )
# print(g)




# # # %%
# # import spacy

# # from pseudo_semantic_graph import TreeNode

# # nlp = spacy.load("en_core_web_trf")
# # nlp.add_pipe("coreferee")

# # # %%
# # t = TreeNode.from_spacy(nlp("He worked in France and Italy."))
# # t.prune_and_merge()
# # t.rearrange()
# # t.visualize()
# # # %%

# # %%
# import spacy

# nlp = spacy.load("en_core_web_trf")
# a1 = nlp(coref_text)
# print(len(a1))
# print(a1[149])
# # %%

# # # %%
# # from spacy import displacy

# # displacy.serve(a1, style="dep")
# # # %%
