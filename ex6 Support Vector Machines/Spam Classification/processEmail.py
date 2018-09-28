#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
import re
from getVocabList import get_vocab_list
from nltk.stem import PorterStemmer


def process_email(email_contents):
    # Load Vocabulary
    vocab_list = get_vocab_list()[:, 1]
    word_indices = []
    # ========================== Preprocess Email ===========================
    # Lower case
    email_contents = str(email_contents)
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub(r'[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)

    # ========================== Tokenize Email ===========================
    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n')

    # Process file
    l_count = 0

    # Tokenize and also get rid of any punctuation
    partition_text = re.split(r'[ @$/#.-:&*+=\[\]?!(){},\'\">_<;%\n\f]', email_contents)

    stemmer = PorterStemmer()

    for one_word in partition_text:
        if one_word != '':
            # Remove any non alphanumeric characters
            one_word = re.sub(r'[^a-zA-Z0-9]', '', one_word)

            # Stem the word
            # (the porterStemmer sometimes has issues, so we use a try catch block)
            one_word = stemmer.stem(one_word)

            # % Skip the word if it is too short.
            if str == '':
                continue

            temp = np.argwhere(vocab_list == one_word)
            if temp.size == 1:
                word_indices.append(temp.min())

            #    % Print to screen, ensuring that the output lines are not too long
            if (l_count + len(one_word) + 1) > 78:
                print('\n')
                l_count = 0
            print('%s' % one_word, end=' ')
            l_count = l_count + len(one_word) + 1
    print('\n')
    # Print footer
    print('\n\n=========================\n')
    return np.array(word_indices)
