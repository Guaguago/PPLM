#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
"""Script for the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2019-ECDT) Task2.
This script evaluates the distinct[1] of the submitted model.
This uses a the version of the dataset which does not contain the "Golden Response" .
Leaderboard scores will be run in the same form but on a hidden test set.

reference:

[1] Li, Jiwei, et al. "A diversity-promoting objective function for neural conversation models."
    arXiv preprint arXiv:1510.03055 (2015).

This requires each team to implement the following function:
def gen_response(self, contexts):
    return a list of responses for each context
    Arguments:
    contexts -- a list of context, each context contains dialogue histories and personal profiles of every speaker
    Returns a list, where each element is the response of the corresponding context
"""
# from main import Model
import json
import sys

# import tokenizer  ## this depends on which model are you currently using
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

import spacy


# spacyspacy_nlp = spacy.load('en_core_web_sm')


def read_dialog(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with open(file) as f:
        contents = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    # return [json.loads(i) for i in contents]
    return contents


def read_responses(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with open(file) as f:
        responses = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    # return [json.loads(i) for i in contents]
    return responses


def count_ngram(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)


def eval_distinct(hyps_resp, n):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: average distinct score for 1, 2-gram
    """

    hyps_resp = [list(map(str, h.split())) for h in hyps_resp]

    # hyps_resp = [list(map(str, tokenizer.encode(h))) for h in hyps_resp]
    # hyps_resp = [list(map(str, [token for token in tokenizer.encode(h) if not token.is_stop])) for h in hyps_resp]

    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    hyps_resp = [(' '.join(i)).split() for i in hyps_resp]
    num_tokens = sum([len(i) for i in hyps_resp])
    num_ngram = count_ngram(hyps_resp, n)
    dist = num_ngram / float(num_tokens)
    # dist2 = count_ngram(hyps_resp, 2) / float(num_tokens)
    # dist3 = count_ngram(hyps_resp, 3) / float(num_tokens)

    # return truncate(dist1, 2), truncate(dist2, 2), truncate(dist3, 2)
    return dist
    # , dist2, dist3


def print_dist_scores(topic, n):
    variants = ['B', 'BR', 'BC', 'BCR']
    locations = ['automated_evaluation/{}/baseline (B)'.format(topic),
                 'automated_evaluation/{}/baseline+reranking (BR)'.format(topic),
                 'automated_evaluation/{}/gradient (BC)'.format(topic),
                 'automated_evaluation/{}/gradient+reranking (BCR)'.format(topic)]
    print('{} distinct-{}:'.format(topic, n))

    for i in range(len(variants)):
        responses = read_responses(locations[i])
        outputs = eval_distinct(responses, n)
        print(outputs)


if __name__ == '__main__':
    print_dist_scores('religion', 2)
