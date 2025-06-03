from ..annotations import (ARGUMENT_LABELS, MODIFIER_LABELS, NOUN_POS,
                           SUBJ_LABELS)


class RearrangeMixin:
    def _redirect_conjunct_arguments(self) -> None:
        """
        Handles conjuncts in argument positions (subjects and objects)
        Example: "He worked in France and Italy."
        """
        for child in self.arguments + self.modifiers:
            child._redirect_conjunct_arguments()

        new_arguments = []
        arguments_to_remove = []
        
        for arg in self.arguments:
            # Process conjuncts in arguments
            conjuncts = []
            remaining_args = []
            
            for sub_arg in arg.arguments:
                if sub_arg.dep == "conj":
                    # Promote the conjunct to same level as original argument
                    sub_arg.dep = arg.dep
                    sub_arg.semantic_role = arg.semantic_role
                    conjuncts.append(sub_arg)
                else:
                    remaining_args.append(sub_arg)
            
            arg.arguments = remaining_args
            new_arguments.extend(conjuncts)
            
            # Check if this argument itself is a conjunct that needs promotion
            if arg.dep == "conj" and self.dep in ARGUMENT_LABELS:
                arguments_to_remove.append(arg)
                new_arguments.append(arg)
        
        self.arguments.extend(new_arguments)
        self.arguments = [a for a in self.arguments if a not in arguments_to_remove]

    def _redirect_conjunct_modifiers(self) -> None:
        """
        Handles conjuncts in modifier positions
        Example: "He worked quickly and efficiently."
        """
        for child in self.arguments + self.modifiers:
            child._redirect_conjunct_modifiers()

        new_modifiers = []
        modifiers_to_remove = []
        
        for mod in self.modifiers:
            # Process conjuncts in modifiers
            conjuncts = []
            remaining_mods = []
            
            for sub_mod in mod.modifiers:
                if sub_mod.dep == "conj":
                    # Promote the conjunct to same level as original modifier
                    sub_mod.dep = mod.dep
                    conjuncts.append(sub_mod)
                else:
                    remaining_mods.append(sub_mod)
            
            mod.modifiers = remaining_mods
            new_modifiers.extend(conjuncts)
            
            # Check if this modifier itself is a conjunct that needs promotion
            if mod.dep == "conj" and self.dep in MODIFIER_LABELS:
                modifiers_to_remove.append(mod)
                new_modifiers.append(mod)
        
        self.modifiers.extend(new_modifiers)
        self.modifiers = [m for m in self.modifiers if m not in modifiers_to_remove]

    def _redirect_attribute_conjuncts(self):
        """
        Example: "He worked in France, Italy and Germany."
        """
        [c._redirect_attribute_conjuncts() for c in self.children]

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

    def _redirect_root_conjuncts(self) -> None:
        """
        Handles root-level conjuncts (coordinated main clauses)
        Example: "He worked in France and lived in Italy."
        """
        if self.dep != "ROOT":
            return

        # Find all conjunct verbs under root
        conjunct_verbs = []
        for arg in self.arguments:
            if arg.dep == "conj" and arg.onto_tag.startswith("VB"):
                conjunct_verbs.append(arg)
        
        if not conjunct_verbs:
            return

        # Find the primary subject
        primary_subjects = [a for a in self.arguments if a.dep in SUBJ_LABELS]
        if not primary_subjects:
            return

        # Redirect each conjunct verb
        for verb in conjunct_verbs:
            # Find the subject for this conjunct (if any)
            conjunct_subj = next(
                (a for a in verb.arguments if a.dep in SUBJ_LABELS),
                None
            )
            
            if not conjunct_subj:
                # If no subject in conjunct, use primary subject
                verb.dep = primary_subjects[0].dep
                primary_subjects[0].arguments.append(verb)
                self.arguments.remove(verb)

    def _merge_prepositional_args(self) -> None:
        """
        Merges prepositional phrases into their semantic roles
        Example: "He was born in France" -> "born [location: France]"
        """
        for child in self.arguments + self.modifiers:
            child._merge_prepositional_args()

        new_arguments = []
        args_to_remove = []
        
        for arg in self.arguments:
            if arg.dep == "prep":
                # Find the object of the preposition
                prep_objects = [a for a in arg.arguments if a.dep in OBJ_LABELS]
                if prep_objects:
                    # Create new semantic role based on preposition
                    prep_obj = prep_objects[0]
                    prep_obj.semantic_role = arg.word[0].lower()  # e.g., "in" -> "location"
                    prep_obj.dep = "prep_obj"
                    new_arguments.append(prep_obj)
                    args_to_remove.append(arg)
        
        self.arguments.extend(new_arguments)
        self.arguments = [a for a in self.arguments if a not in args_to_remove]

    def rearrange(self) -> None:
        """
        Main method to reorganize the tree structure for better semantic representation
        """
        # Process children first (post-order traversal)
        for child in self.arguments + self.modifiers:
            child.rearrange()

        # Handle different types of rearrangements
        self._redirect_conjunct_arguments()
        self._redirect_conjunct_modifiers()
        self._redirect_root_conjuncts()
        self._merge_prepositional_args()