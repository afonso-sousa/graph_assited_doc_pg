import itertools
from typing import List

from ..annotations import (CONTENT_POS, IGNORED_LABELS, MODIFIER_LABELS,
                           MONTHS, PREP_POS, PRESERVE_LABELS, PRONOUN_POS,
                           QUANTIFIER_POS, WH_POS)


class PruneAndMergeMixin:
    def met_pruning_conditions(self, node: "TreeNode") -> bool:
        """Determine if a node should be pruned from the semantic graph."""
        # Keep all content words
        if node.onto_tag in CONTENT_POS:
            return False
          
        # Keep pronouns
        if node.onto_tag in PRONOUN_POS:
            return False
       
        # Keep wh-words
        if node.onto_tag in WH_POS:
            return False
        
        # Keep numbers and quantifiers unless they're simple determiners
        if node.onto_tag in QUANTIFIER_POS and node.dep != "det":
            return False
        
        # Check if this is a label we generally ignore
        if node.dep in IGNORED_LABELS:
            # But preserve special cases
            if node.dep in PRESERVE_LABELS:
                return False
            return True
        
        # Keep prepositions as they often indicate semantic relations
        if node.onto_tag in PREP_POS:
            return False
        
        # Default: keep the node if we're unsure
        return False

    def handle_negation(self, candidates_to_merge: List["TreeNode"]) -> List["TreeNode"]:
        """Handle negation words in the modifiers to be merged."""
        # Find and standardize negation markers
        for modifier in candidates_to_merge:
            if modifier.dep == "neg":
                modifier.word = ["not"]  # Standardize negation representation
        
        # Remove auxiliary verbs if we found a negation (no longer needed as we're keeping auxiliaries)
        return candidates_to_merge

    def merge_modifiers(self) -> None:
        """Merge adjacent modifiers into the current node when appropriate."""
        if not self.modifiers:
            return

        # Find modifiers that meet our criteria for merging
        candidates_to_merge = [
            modifier
            for modifier in self.modifiers
            if not modifier.arguments and not modifier.modifiers  # Is leaf node
            and modifier.dep in MODIFIER_LABELS  # Has modifier dependency type
            and modifier.onto_tag in CONTENT_POS.union({"RB", "IN", "TO"})  # Content words or prepositions
            and not any(word.lower() in MONTHS for word in modifier.word)  # Not a month name
        ]

        if not candidates_to_merge:
            return

        remaining_modifiers = [
            modifier for modifier in self.modifiers 
            if modifier not in candidates_to_merge
        ]

        candidates_to_merge = self.handle_negation(
            candidates_to_merge
        )

        # Combine words and indices in sentence order
        all_elements = [(mod.word, mod.index) for mod in candidates_to_merge] + [(self.word, self.index)]
        sorted_elements = sorted(all_elements, key=lambda x: min(x[1]))
        
        # Flatten and update the current node
        self.word = list(itertools.chain(*[words for words, _ in sorted_elements]))
        self.index = list(itertools.chain(*[indices for _, indices in sorted_elements]))
        self.modifiers = remaining_modifiers

    def prune_and_merge(self) -> None:
        """Prune unnecessary nodes and merge adjacent modifiers."""
        # Process children first (post-order traversal)
        for child in self.arguments + self.modifiers:
            child.prune_and_merge()

        # Prune arguments
        self.arguments = [
            arg for arg in self.arguments 
            if not self.met_pruning_conditions(arg)
        ]

        # Prune modifiers
        self.modifiers = [
            mod for mod in self.modifiers 
            if not self.met_pruning_conditions(mod)
        ]

        # Merge adjacent modifiers
        self.merge_modifiers()
