import pandas as pd
import numpy as np
from collections import Counter
import argparse
from collections import defaultdict
import math
import sys

def read_interests_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

def count_words_in_interests(interests, lower=True):
    word_counts = defaultdict(int)
    for interest in interests:
        if lower:
            unique_words = set(interest.lower().split())
        else:
            set(interest.split())

        for word in unique_words:
            word_counts[word] += 1
    return word_counts


def get_log_odds(file1, file2, file0, verbose=False, lower=True):
    interests1 = read_interests_from_file(file1)
    interests2 = read_interests_from_file(file2)

    counts1 = count_words_in_interests(interests1, lower)
    counts2 = count_words_in_interests(interests2, lower)

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    n1 = len(interests1)
    n2 = len(interests2)

    all_words = set(counts1.keys()).union(counts2.keys())
    
    for word in all_words:
        counts1[word] = counts1.get(word, 0) + 0.5
        counts2[word] = counts2.get(word, 0) + 0.5
    
    for word in all_words:
        l1 = counts1[word] / (n1  - counts1[word])
        l2 = counts2[word] / (n2  - counts2[word])
        sigmasquared[word] = 1 /counts1[word] + 1 /counts2[word]
        sigma[word] = math.sqrt(sigmasquared[word])
        delta[word] = (math.log(l1) - math.log(l2)) / sigma[word]

    if verbose:
        for word in sorted(delta, key=delta.get)[:10]:
            print("%s, %.3f" % (word, delta[word]))
        for word in sorted(delta, key=delta.get, reverse=True)[:10]:
            print("%s, %.3f" % (word, delta[word]))

    return delta
    

