from ..annotations import SUBJ_LABELS


class RearrangeMixin:
    def _promote_conj_children(self) -> None:
        """
        Traverse the node's children (arguments + modifiers),
        and promote any 'conj' child to be a sibling of its parent.
        """
        # Combine children from all possible subnode types
        all_children = getattr(self, "arguments", []) + getattr(self, "modifiers", [])
        promoted = []

        for child in all_children:
            # Recurse before processing to ensure bottom-up traversal
            child._promote_conj_children()

            for rel_name in ["arguments", "modifiers"]:
                subnodes = getattr(child, rel_name, [])
                kept = []
                for sub in subnodes:
                    if sub.dep == "conj":
                        # Promote to same level as its parent
                        sub.dep = child.dep
                        if rel_name == "arguments":
                            sub.semantic_role = getattr(child, "semantic_role", None)
                        promoted.append(sub)
                    else:
                        kept.append(sub)
                setattr(child, rel_name, kept)

        # Reattach promoted conjuncts to this node
        if hasattr(self, "arguments"):
            self.arguments.extend([p for p in promoted if p.dep in SUBJ_LABELS or p.onto_tag.startswith("VB")])
        if hasattr(self, "modifiers"):
            self.modifiers.extend([p for p in promoted if p.dep not in SUBJ_LABELS and not p.onto_tag.startswith("VB")])

    def _promote_conj_verb_clauses(self) -> None:
        """
        Promote all conjunct verbs under a root or clausal verb to the same level,
        attaching them to the main subject if needed, and removing 'conj' edges.
        """
        if self.dep != "ROOT" and not self.onto_tag.startswith("VB"):
            return

        # Determine anchor subject
        subject_nodes = [a for a in getattr(self, "arguments", []) if a.dep in SUBJ_LABELS]
        if not subject_nodes:
            return
        anchor_subject = subject_nodes[0]

        # Search for conjunct verb children in arguments and modifiers
        for container in [self.arguments, self.modifiers]:
            to_promote = []
            for node in container:
                if node.dep == "conj" and node.onto_tag.startswith("VB"):
                    # Check if the conj-verb already has its own subject
                    has_subject = any(
                        child.dep in SUBJ_LABELS
                        for child in getattr(node, "arguments", []) + getattr(node, "modifiers", [])
                    )
                    if not has_subject:
                        node.dep = anchor_subject.dep
                        anchor_subject.arguments.append(node)
                        to_promote.append(node)
                    else:
                        node.dep = "ROOT"

            # Remove promoted verbs from the original container
            for node in to_promote:
                container.remove(node)

    def rearrange(self) -> None:
        """
        Main method to promote conjuncts and reroute subjects of root-level verbs.
        """
        self._promote_conj_children()
        self._promote_conj_verb_clauses()
