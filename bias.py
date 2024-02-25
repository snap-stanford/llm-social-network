# adapted from: source https://github.com/jmhessel/FightingWords/blob/master/fighting_words_py3.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as CV
import string
exclude = set(string.punctuation)
import os
import json

from constants_and_utils import *

def basic_sanitize(in_string):
    '''Returns a very roughly sanitized version of the input string.'''
    in_string = in_string.replace(" and ", " ")
    in_string = in_string.replace(" or ", " ")
    in_string = ''.join([ch for ch in in_string if ch not in exclude])
    in_string = in_string.lower()
    in_string = ' '.join(in_string.split())

    return in_string

def bayes_compare_language(l1, l2, ngram = 1, prior=.05, cv = None):
    '''
    Arguments:
    - l1, l2; a list of strings from each language sample
    - ngram; an int describing up to what n gram you want to consider (1 is unigrams,
    2 is bigrams + unigrams, etc). Ignored if a custom CountVectorizer is passed.
    - prior; either a float describing a uniform prior, or a vector describing a prior
    over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
    when you make your CountVectorizer object.
    - cv; a sklearn.feature_extraction.text.CountVectorizer object, if desired.

    Returns:
    - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
    if cv is None and type(prior) is not float:
        print("If using a non-uniform prior:")
        print("Please also pass a count vectorizer with the vocabulary parameter set.")
        quit()

    # this is some basic normalization, not tokenization, we may want to change that
    l1 = [basic_sanitize(l) for l in l1]
    l2 = [basic_sanitize(l) for l in l2]

    # I removed max_df, and min_df from CV ()
    # let's think if we want to have it back and what values

    if cv is None:
        cv = CV(decode_error = 'ignore', ngram_range=(1,ngram),
                binary = False,
                max_features = 15000)
    counts_mat = cv.fit_transform(l1+l2).toarray()

    """
    for the example below it returns such a matrix
    l1 = ["dancing", "dancing", "dancing", "sport", "ballet", "swimming", "running", "hiking"]
    l2 = ["singing", "singing", "singing", "traveling", "thinking", "talking", "chatting"]
    >>> print(counts_mat)
    [[0 0 1 0 0 0 0 0 0 0 0]
    [0 0 1 0 0 0 0 0 0 0 0]
    [0 0 1 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 1 0 0 0 0]
    [1 0 0 0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 1 0 0 0]
    [0 0 0 0 1 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0 0 0 0]
    [0 0 0 0 0 1 0 0 0 0 0]
    [0 0 0 0 0 1 0 0 0 0 0]
    [0 0 0 0 0 1 0 0 0 0 0]
    [0 0 0 0 0 0 0 0 0 0 1]
    [0 0 0 0 0 0 0 0 0 1 0]
    [0 0 0 0 0 0 0 0 1 0 0]
    [0 1 0 0 0 0 0 0 0 0 0]]
    """

    # Now sum over languages...
    vocab_size = len(cv.vocabulary_)
    #print("Vocab size is {}".format(vocab_size))

    # if we want to use informative prior we need to give it as an argument
    # (but we can leave noninformative prior as well)
    if type(prior) is float:
        priors = np.array([prior for i in range(vocab_size)])
    else:
        priors = prior

    z_scores = np.empty(priors.shape[0])
    count_matrix = np.empty([2, vocab_size], dtype=np.float32)
    # top of the matrix is first document, below is second document
    # we put it in count_matrix[0,:] and [1, :] respectively
    # Please see the explanation below
    count_matrix[0, :] = np.sum(counts_mat[:len(l1), :], axis = 0)
    count_matrix[1, :] = np.sum(counts_mat[len(l1):, :], axis = 0)
    a0 = np.sum(priors)
    n1 = 1.*np.sum(count_matrix[0,:]) # sum of words in doc1
    n2 = 1.*np.sum(count_matrix[1,:]) # sum of words in doc2

    for i in range(vocab_size):
        #compute delta
        # (count of word i in doc 0 + prior word i) / (# words in doc 0 + total prior on words - cound of word i in doc 0 - prior word i)
        term1 = np.log((count_matrix[0,i] + priors[i])/(n1 + a0 - count_matrix[0,i] - priors[i]))
        term2 = np.log((count_matrix[1,i] + priors[i])/(n2 + a0 - count_matrix[1,i] - priors[i]))
        delta = term1 - term2
        #compute variance on delta
        # this formula is an approximation; please see the variance section
        var = 1./(count_matrix[0,i] + priors[i]) + 1./(count_matrix[1,i] + priors[i])
        #store final score
        z_scores[i] = delta/np.sqrt(var)
    index_to_term = {v:k for k,v in cv.vocabulary_.items()}
    sorted_indices = np.argsort(z_scores)
    return_list = []
    for i in sorted_indices:
        return_list.append((index_to_term[i], z_scores[i]))
    return return_list


if __name__ == "__main__":
    # Example call:
    # l1 = ["dancing", "dancing", "dancing", "sport", "ballet", "swimming", "running", "hiking"]
    # l2 = ["singing", "singing", "singing", "traveling", "thinking", "talking", "chatting"]
    # print(bayes_compare_language(l1, l2, ngram = 1, prior=.01, cv = None))

    # read path to text files / jsons
    with open(os.path.join(PATH_TO_TEXT_FILES, 'us_5000_with_interests.json')) as f:
        personas = json.load(f)

    # get all interests
    all_interests = []
    for persona in personas:
        all_interests.append(personas[persona]["interests"])

    demos = {'gender': ['Woman', 'Man', 'Nonbinary'],
             'race/ethnicity': ['White', 'Black', 'Latino', 'Asian', 'Native American/Alaska Native', 'Native Hawaiian'],
             'religion': ['Protestant', 'Catholic', 'Jewish', 'Muslim', 'Hindu', 'Buddhist', 'Unreligious'],
             'political affiliation': ['Democrat', 'Republican', 'Independent']}

    for demo in demos:
        for category  in demos[demo]:
            group_count = 0
            demo_interests = []
            for persona in personas:
                if personas[persona][demo] == category:
                    demo_interests.append(personas[persona]["interests"])
                    group_count += 1
            print("\n\n")
            print(f"Comparing interests for {demo} {category}...")
            print(f"Group count: {group_count/len(personas)*100}%")
            z_scores = bayes_compare_language(demo_interests, all_interests, ngram = 1, prior=.01, cv = None)
            # print("NOT interested in")
            # print("-------------------------")
            # for (word, z) in z_scores:
            #     if z < -1.96:
            #         print(f"{word},", end =" ")

            z_scores.reverse()
            print("\nInterested in")
            print("-------------------------")
            for (word, z) in z_scores:
                if z > 1.96:
                    print(f"{word},", end =" ")
            print("\n\n")
    pass