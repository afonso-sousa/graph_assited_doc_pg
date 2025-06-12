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

    def _redirect_root_conj_verb_subjects(self) -> None:
        """
        Promote verb conjuncts under ROOT and attach them to the main subject if they lack their own.
        E.g., "He worked in France and lived in Italy" -> 'lived' should be attached under 'He'
        """
        if self.dep != "ROOT":
            return

        # Find all subject(s) under root
        primary_subjects = [arg for arg in getattr(self, "arguments", []) if arg.dep in SUBJ_LABELS]
        breakpoint()
        if not primary_subjects:
            return

        primary_subject = primary_subjects[0]

        conjunct_verbs = []
        for node in getattr(self, "arguments", []):
            if node.dep == "conj" and node.onto_tag.startswith("VB"):
                conjunct_verbs.append((node, "arguments"))
        for node in getattr(self, "modifiers", []):
            if node.dep == "conj" and node.onto_tag.startswith("VB"):
                conjunct_verbs.append((node, "modifiers"))

        for node, container_name in conjunct_verbs:
            has_subject = any(a.dep in SUBJ_LABELS for a in getattr(node, "arguments", []))
            if not has_subject:
                node.dep = primary_subject.dep
                primary_subject.arguments.append(node)

                # Remove from original container
                container = getattr(self, container_name)
                container.remove(node)

    def rearrange(self) -> None:
        """
        Main method to promote conjuncts and reroute subjects of root-level verbs.
        """
        self._promote_conj_children()
        self._redirect_root_conj_verb_subjects()
