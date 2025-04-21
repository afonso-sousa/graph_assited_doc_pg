# %%
from itertools import chain

import spacy
from nltk.corpus import wordnet

nlp = spacy.load("en_core_web_lg")


def filter_unwanted(word, syn):
    if syn.find("_") > -1:
        return False

    if word.casefold() == syn.casefold():
        return False

    return True


def generate_syn_candidates(word):
    syns = []
    hypothesis_syns = set(
        chain.from_iterable(
            (lemma.name() for lemma in synset.lemmas())
            for synset in wordnet.synsets(word)
        )
    )
    filtered_hypothesis_syns = list(
        filter(lambda x: filter_unwanted(word, x), hypothesis_syns)
    )
    for syn in filtered_hypothesis_syns:
        sim_score = nlp(word).similarity(nlp(syn))
        syns.append((round(sim_score, 3), syn))
    syns = sorted(syns, reverse=True)

    return syns


def match_case(word, new_word):
    if word.isupper():
        new_word = new_word.upper()
    elif word.islower():
        new_word = new_word.lower()
    elif word.istitle():
        new_word = new_word.title()
    return new_word


def match_pos(word, new_word):
    word = nlp(word)[0]
    new_word = nlp(new_word)[0]
    return word.pos_ == new_word.pos_ or word.tag_ == new_word.tag_


def find_synonym(word, similarity_threshold=0.5):
    syn_candidates = generate_syn_candidates(word.lower())
    best_candidate = None
    best_similarity = 0
    for sim, syn in syn_candidates:
        if (
            sim > best_similarity
            and sim > similarity_threshold
            and match_pos(word, syn)
        ):
            best_candidate = match_case(word, syn)
            best_similarity = sim

    return best_candidate


def generate_ant_candidates(word):
    hypothesis_ants = set(
        antonym.name()
        for synset in wordnet.synsets(word)
        for lemma in synset.lemmas()
        for antonym in lemma.antonyms()
    )
    filtered_hypothesis_ants = list(
        filter(lambda x: filter_unwanted(word, x), hypothesis_ants)
    )

    return filtered_hypothesis_ants


def find_antonym(word):
    ant_candidates = generate_ant_candidates(word.lower())
    for ant in ant_candidates:
        if match_pos(word, ant):
            return ant

    return None
