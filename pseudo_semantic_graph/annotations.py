from typing import Final, List

PRUNE_LIST = ["punct", "cc", "preconj", "mark", "case", "det"]

# Penn TreeBank POS
VERB_POS: Final[List[str]] = [
    "VBZ",  # Verb, 3rd person singular present
    "VBN",  # Verb, past participle
    "VBD",  # Verb, past tense
    "VBP",  # Verb, non-3rd person singular present
    "VB",  # Verb, base form
    "VBG",  # Verb, gerund or present participle
]

NOUN_POS: Final[List[str]] = ["NN", "NNP", "NNS", "NNPS"]

# Stanford typed dependencies
SUBJ_LABELS: Final[List[str]] = [
    "nsubj",  # nominal subject
    "nsubjpass",  # passive nominal subject
    "csubj",  # clausal subject
    "csubjpass",  # clausal passive subject
]
OBJ_LABELS: Final[List[str]] = [
    "dobj",  # direct object
    "pobj",  # object of a preposition
    "iobj",  # indirect object
]
SUBJ_AND_OBJ: Final[List[str]] = SUBJ_LABELS + OBJ_LABELS

# CONJ: Final[List[str]] = ["conj", "parataxis"]
CONJ: Final[List[str]] = ["conj", "cc", "preconj", "parataxis"]

PREP_POS: Final[List[str]] = ["PP", "IN", "TO"]

MODIFIER_POS: Final[List[str]] = [
    "JJ",
    "FW",
    "JJR",
    "JJS",
    "RB",
    "RBR",
    "RBS",
    "WRB",
    # "CD",
]
MODIFIERS: Final[List[str]] = [
    "amod",
    "nn",
    "mwe",
    "num",
    "quantmod",
    "dep",
    "number",
    "auxpass",
    "partmod",
    # "poss",
    "possessive",
    "neg",
    "advmod",
    # "npadvmod",
    "advcl",
    "aux",
    "det",
    "predet",
    # "appos",
]
ADDITIONAL_TO_MERGE: Final[List[str]] = [
    # "nummod",
    "nmod",
    "punct",
    "compound",
]
MODIFIERS_PLUS: Final[List[str]] = MODIFIERS + ADDITIONAL_TO_MERGE

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

RELATIONS_MAPPING = {
    **dict.fromkeys(SUBJ_LABELS, "ARG0"),
    **dict.fromkeys(OBJ_LABELS, "ARG1"),
    "acomp": "ARG2",
    # "dative": "to",
    # "poss": "of",
}
