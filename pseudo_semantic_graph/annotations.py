from typing import Final, List, Set

MONTHS: Final[List[str]] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

# Content word POS tags (should generally be preserved)
CONTENT_POS: Final[Set[str]] = {
    # Nouns
    "NN", "NNS", "NNP", "NNPS",
    # Verbs
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    # Adjectives
    "JJ", "JJR", "JJS",
    # Adverbs
    "RB", "RBR", "RBS"
}

# Noun POS tags (subset of CONTENT_POS)
NOUN_POS: Final[Set[str]] = {
    "NN", "NNS", "NNP", "NNPS",  # Regular nouns
    "PRP", "PRP$",               # Pronouns
    "CD"                         # Numbers (often behave like nouns)
}

# Pronouns (important for coreference)
PRONOUN_POS: Final[Set[str]] = {"PRP", "PRP$"}

VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
NOUN_TAGS = {"NN", "NNS"}  # subset of NOUN_POS for visualization
PROPN_TAGS = {"NNP", "NNPS"}

# WH-words (important for questions and relative clauses)
WH_POS: Final[Set[str]] = {"WP", "WP$", "WRB", "WDT"}

# Numbers and quantifiers
QUANTIFIER_POS: Final[Set[str]] = {"CD"}

# Prepositions and particles
PREP_POS: Final[Set[str]] = {"IN", "TO"}

# Argument-like dependencies
SUBJ_LABELS: Final[List[str]] = [
    "nsubj", "nsubjpass", "csubj", "csubjpass", "expl"
]
OBJ_LABELS: Final[List[str]] = [
    "dobj", "iobj", "obj", "pobj", "attr", "oprd"
]
CLAUSAL_COMPLEMENTS: Final[List[str]] = [
    "ccomp", "xcomp"
]
CONJ: Final[List[str]] = [
    "conj", "cc", "preconj", "parataxis"
]

ARGUMENT_LABELS: Final[List[str]] = SUBJ_LABELS + OBJ_LABELS + CLAUSAL_COMPLEMENTS + ["agent"]

# Modifier-like dependencies
MODIFIER_LABELS: Final[List[str]] = [
    "amod", "advmod", "nmod", "appos", "acl", "advcl", "relcl",
    "poss", "prep", "quantmod", "npadvmod", "tmod", "prt"
]

# Dependencies to ignore (often function words or non-content)
IGNORED_LABELS: Final[List[str]] = [
    "punct", "aux", "cop", "cc", "mark", "case", "det", "auxpass"
]

# Special cases that should be preserved despite being in IGNORED_LABELS
PRESERVE_LABELS: Final[Set[str]] = {"neg"}  # Negation markers

RELATIONS_MAPPING = {
    **dict.fromkeys(SUBJ_LABELS, "ARG0"),
    **dict.fromkeys(OBJ_LABELS, "ARG1"),
    "acomp": "ARG2",
    "xcomp": "ARG2",
    "ccomp": "ARG2",
    "advcl": "ARGM",     # Adverbial clause
    "advmod": "ARGM",    # Adverbial modifier
    "amod": "MOD",       # Adjective modifier
    "nmod": "MOD",       # Noun modifier
    "poss": "poss",      # Possessive (could be 'of' or ARG0 depending)
    "prep": "prep",      # Prepositional link
    "pobj": "ARG1",      # Often object of preposition, same as OBJ
    "appos": "MOD",      # Appositional modifier
    "compound": "MOD",   # Noun compound
    "nn": "MOD",         # Legacy noun compound
    "conj": "CONJ",      # Coordinated conjunction
    "cc": "CONJ",        # Conjunction
    "attr": "ARG1",      # Attribute (copula predicate)
    "relcl": "ARGM",     # Relative clause
}