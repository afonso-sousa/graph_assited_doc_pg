import itertools

from ..annotations import *


class PruneAndMergeMixin:
    def met_pruning_conditions(self, node: "TreeNode") -> bool:
        return (
            node.onto_tag not in PREP_POS
            and (node.dep in PRUNE_LIST or node.ud_pos == "AUX")
            and node.word != ["-"]
        )

    def handle_negation(self, candidates_to_join, remaining_attributes):
        found_negation = False
        for attribute in candidates_to_join:
            if attribute.dep == "neg":
                found_negation = True
                attribute.word = ["not"]
        if found_negation:
            for attribute in candidates_to_join:
                if attribute.ud_pos == "AUX" and attribute.onto_tag == "VBD":
                    candidates_to_join.remove(attribute)

        return candidates_to_join, remaining_attributes

    def merge_nodes(self) -> None:
        if not self.attributes:
            return

        candidates_to_join = [
            attribute
            for attribute in self.attributes
            if (not attribute.children)  # is leaf node
            and (
                (attribute.dep in MODIFIERS_PLUS or attribute.onto_tag in MODIFIER_POS)
                and not attribute.onto_tag == "WRB"
            )  # has the syntactic type we want to merge
            and not any(word.lower() in MONTHS for word in attribute.word)
        ]
        if not candidates_to_join:
            return

        remaining_attributes = [
            attribute
            for attribute in self.attributes
            if attribute not in candidates_to_join
        ]

        candidates_to_join, remaining_attributes = self.handle_negation(
            candidates_to_join, remaining_attributes
        )

        word_list, index_list = zip(
            *[(candidate.word, candidate.index) for candidate in candidates_to_join]
            + [(self.word, self.index)]
        )

        sorted_pairs = sorted(zip(word_list, index_list), key=lambda pair: min(pair[1]))

        sorted_words, sorted_indices = list(zip(*sorted_pairs))

        self.word = list(itertools.chain(*sorted_words))
        self.index = list(itertools.chain(*sorted_indices))
        self.attributes = remaining_attributes

    def prune_and_merge(self) -> None:
        for child in self.children:
            if self.met_pruning_conditions(child):
                # redirect children of pruned nodes to grandparents
                self.add_nouns(child.nouns)
                self.add_verbs(child.verbs)
                self.add_attributes(child.attributes)
                self.remove_child(child)

        [c.prune_and_merge() for c in self.children]

        self.merge_nodes()
