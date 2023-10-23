#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter 

#read in interests as discreet interests and not large block of texts
def interests_from_text_file(fn):
    try:
        with open(fn, 'r') as f:
            interests = [line.strip() for line in f.readlines()]
        return interests
    except FileNotFoundError:
        print(f"File not found: {fn}")
        return None

# tokenizes each word, lowercases all, removes punctuation. has option to include or exclude stop words
def preprocess_all_interests(interests, remove_stop_words=False):
    preprocessed_interests = []
    for interest in interests:
        words = word_tokenize(interest)
        words = [word.lower() for word in words]
        words = [word for word in words if word.isalnum()]

        if remove_stop_words:
            stop_words = set(stopwords.words("english"))
            words = [word for word in words if word not in stop_words]
            
        preprocessed_interests.append(words)
    return preprocessed_interests

#TODO: Complete prior calculation to weight log odds ratio
def dirichlet_prior(word_counts):
    return 0

def log_odds_ratio_by_total_words(word_counts):
    l_o_rs = {}
    total_word_count = sum(word_counts.values())
    for word, count in word_counts.items():
        l_o_r = np.log((count + 1e-9) / (total_word_count - count + 1e-9))
        l_o_rs[word] = l_o_r
    return l_o_rs


def log_odds_ratio_by_interests(preprocessed_interests):
    l_o_rs = {}
    total_interest_count = len(preprocessed_interests)
    interest_word_counts = Counter()
    
    for preprocessed_interest in preprocessed_interests:
        unique_words = set(preprocessed_interest)
        interest_word_counts.update(unique_words)

    for word, count in interest_word_counts.items():
        l_o_r = np.log((count + 1e-9) / (total_interest_count - count + 1e-9))
        l_o_rs[word] = l_o_r

    return l_o_rs

def filter_significant_log_odds_ratios(log_odds_ratios):
    values = list(log_odds_ratios.values())
    
    mean_lor = np.mean(values)
    std_lor = np.std(values)
    
    significant_lors = {}
    for word, lor in log_odds_ratios.items():
        if np.abs(lor - mean_lor) > 2*std_lor:  
            significant_lors[word] = lor
            
    return significant_lors


if __name__ == "__main__":
    # read interests from a text file
    unmarkedInterests = interests_from_text_file("/Users/mayajosifovska/Desktop/unmarkedInterests.txt")
    
    # preprocess the text
    preprocessed_unmarked_interests = preprocess_all_interests(unmarkedInterests)
    
    # combine all words from all interests into a single list
    all_unmarked_words = [word for sublist in preprocessed_unmarked_interests for word in sublist]
    
    # count occurrences of each word
    unmarked_word_counts = Counter(all_unmarked_words)

    # calculate log odds ratio based on total words
    lor_unmarked_by_total_words = log_odds_ratio_by_total_words(unmarked_word_counts)
    
    # calculate log odds ratio based on the number of interests that a word appears in
    lor_unmarked_by_interests = log_odds_ratio_by_interests(preprocessed_unmarked_interests)
    
   # filter significant log odds ratios based on total words
    significant_lors_by_total_words = filter_significant_log_odds_ratios(lor_unmarked_by_total_words)
    
    # filter significant log odds ratios based on interests
    significant_lors_by_interests = filter_significant_log_odds_ratios(lor_unmarked_by_interests)
    
    # extract only the significant words 
    significant_words_by_total_words = list(significant_lors_by_total_words.keys())
    significant_words_by_interests = list(significant_lors_by_interests.keys())
    
    print("Significant Words by Total Words:", significant_words_by_total_words, '\n')
    print("Significant Words by Interests:", significant_words_by_interests, '\n')


# In[ ]:





# In[ ]:




