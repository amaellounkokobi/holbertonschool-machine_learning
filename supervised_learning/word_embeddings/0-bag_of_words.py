#!/usr/bin/env python3
"""
This module contains a 
function that creates a bag of words embedding matrix

Function:
    def bag_of_words(sentences, vocab=None):
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab:  list of the vocabulary words to use 
        for the analysis
    """

    vocab_list = []
    reg1 = "'s"
    reg2 = "[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]"
    embedding = []
    if vocab is None:
        # Sentences to vocab
        for sentence in sentences:
            sentence = norm_sentence(reg1, reg2, sentence)
            vocab_list += sentence.split()

        vocab_list = [v.lower() for v in vocab_list]
        features = sorted(set(vocab_list))
    else:
        features = vocab

    # Sentences and vocab to embedding
    for sentence in sentences:
        sentence = norm_sentence(reg1, reg2, sentence)
        words_sentence = sentence.split()
        words_sentence = [v.lower() for v in words_sentence]
        count_vocab = []
        for word in features:
            c_word = words_sentence.count(word)
            count_vocab.append(c_word)
        embedding.append(count_vocab)

    return np.array(embedding), features


def norm_sentence(reg1, reg2, sentence):
    """
    Return a sentece without 
    """
    sentence = re.sub(reg1, '', sentence)
    sentence = re.sub(reg2, '', sentence)
    return sentence
