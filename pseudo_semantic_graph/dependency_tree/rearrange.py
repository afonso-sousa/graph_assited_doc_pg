from ..annotations import *


class RearrangeMixin:
    def redirect_nominal_conjuncts(self):
        """
        Example: "He worked in France and Italy."
        """
        [c.redirect_nominal_conjuncts() for c in self.children]

        if not self.nouns:
            return

        nouns_to_rearrange = []
        for noun in self.nouns:
            if noun.dep in SUBJ_AND_OBJ and not noun.verbs:
                current_nouns = []
                for grandchild in noun.nouns:
                    if grandchild.dep == "conj":
                        # inherit new sibling relation
                        grandchild.dep = noun.dep
                        nouns_to_rearrange.append(grandchild)
                    else:
                        current_nouns.append(grandchild)
                noun.nouns = current_nouns

        self.nouns.extend(nouns_to_rearrange)

    def redirect_verbal_conjuncts(self):
        """
        Example: "White was born in PEI, educated in Halifax, and lived in Toronto."
        """
        [c.redirect_verbal_conjuncts() for c in self.children]

        if not self.verbs:
            return

        verbs_to_rearrange = []
        for verb in self.verbs:
            current_verbs = []
            for grandchild in verb.verbs:
                if grandchild.dep == "conj":
                    grandchild.dep = verb.dep
                    verbs_to_rearrange.append(grandchild)
                else:
                    current_verbs.append(grandchild)
            verb.verbs = current_verbs

        self.verbs.extend(verbs_to_rearrange)

    def redirect_attribute_conjuncts(self):
        """
        Example: "He worked in France, Italy and Germany."
        """
        [c.redirect_attribute_conjuncts() for c in self.children]

        if not self.nouns:
            return

        nouns_to_rearrange = []
        for noun in self.nouns:
            if not noun.verbs:
                current_nouns = []
                for grandchild in noun.attributes:
                    if grandchild.dep == "conj" and grandchild.onto_tag in NOUN_POS:
                        grandchild.dep = noun.dep
                        nouns_to_rearrange.append(grandchild)
                    else:
                        current_nouns.append(grandchild)
                noun.attributes = current_nouns

        self.nouns.extend(nouns_to_rearrange)

    def redirect_root_conj_children(self):
        """
        Example: "White was born in PEI, educated in Halifax, and lived in Toronto."
        """
        if self.dep != "ROOT":
            return

        if not self.verbs:
            return
        subject_children = [c for c in self.nouns if c.dep in SUBJ_LABELS]
        if not subject_children:
            return

        verbs_to_remove = []
        for verb in self.verbs:
            if not any(n.dep in SUBJ_LABELS for n in verb.nouns):
                # TODO: can there be more than one subject?
                verb.dep = subject_children[0].dep
                subject_children[0].verbs.append(verb)
                verbs_to_remove.append(verb)

        self.verbs = [v for v in self.verbs if v not in verbs_to_remove]
        # self.verbs.clear()

    def merge_prepositions(self):
        """
        Example: "White was born in PEI, educated in Halifax, and lived in Toronto."
        """
        [c.merge_prepositions() for c in self.children]

        if not self.attributes:
            return

        attributes_to_maintain = []
        nouns_to_rearrange = []
        for attribute in self.attributes:
            if attribute.dep in ["prep", "agent", "dative"]:
                for grandchild in attribute.nouns:
                    grandchild.dep = attribute.word[0].lower()
                    nouns_to_rearrange.append(grandchild)
                attribute.nouns.clear()
            else:
                attributes_to_maintain.append(attribute)

        self.nouns.extend(nouns_to_rearrange)
        self.attributes = attributes_to_maintain
