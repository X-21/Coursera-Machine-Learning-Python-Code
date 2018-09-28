#!/usr/bin/env python
# -*- coding=utf-8 -*-


import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC

from processEmail import process_email
from emailFeatures import email_features
from getVocabList import get_vocab_list


def pause_func():
    while input() != '':
        pass


def load_mat_file(_filename):
    return loadmat(_filename)


def predict_email_spam(_filename):
    _file_contents = open('../data/' + _filename).read()
    _word_indices = process_email(_file_contents)
    _features = email_features(_word_indices)
    _p = Classification.predict(_features.T)
    print('\nProcessed %s\n\nSpam Classification: %d\n' % (_filename, _p[0]))
    print('(1 indicates spam, 0 indicates not spam)\n\n')


if __name__ == '__main__':
    # ==================== Part 1: Email Preprocessing ====================
    # To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
    # to convert each email into a vector of features. In this part, you will
    # implement the preprocessing steps for each email. You should
    # complete the code in processEmail.m to produce a word indices vector
    # for a given email.

    print('\nPreprocessing sample email (emailSample1.txt)\n')
    # Extract Features
    file_contents = open('../data/emailSample1.txt').read()
    word_indices = process_email(file_contents)
    # Print Stats
    print('Word Indices:\n')
    print_index = 0
    for print_value in word_indices:
        print_index += 1
        print("%4d" % print_value, end=' ')
        if print_index % 10 == 0:
            print('\n')
    print('\n')
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ==================== Part 2: Feature Extraction ====================
    # Now, you will convert each email into a vector of features in R^n.
    # You should complete the code in emailFeatures.m to produce a feature
    # vector for a given email.

    print('\nExtracting features from sample email (emailSample1.txt)\n')
    # Extract Features
    features = email_features(word_indices)
    # Print Stats
    print('Length of feature vector: %d\n' % features.size)
    print('Number of non-zero entries: %d\n' % np.sum((features > 0).astype(np.int32)))
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =========== Part 3: Train Linear SVM for Spam Classification ========
    # In this section, you will train a linear classifier to determine if an
    # email is Spam or Not-Spam.

    # Load the Spam Email dataset
    data = load_mat_file('../data/spamTrain.mat')
    print('\nTraining Linear SVM (Spam Classification)\n')
    print('(this may take 1 to 2 minutes) ...\n')
    X = data['X']
    y = data['y'].ravel()
    C = 0.1
    Classification = SVC(C=C, kernel='linear')
    Classification.fit(X, y)
    p = Classification.predict(X)
    print('Training Accuracy: {:.2f}\n'.format((np.mean((p == y)) * 100)))
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =================== Part 4: Test Spam Classification ================
    data = load_mat_file('../data/spamTest.mat')
    Xtest = data['Xtest']
    ytest = data['ytest'].ravel()
    p = Classification.predict(Xtest)
    print('Test Accuracy: {:.2f}\n'.format((np.mean((p == ytest)) * 100)))
    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # ================= Part 5: Top Predictors of Spam ====================
    index_array = np.argsort(Classification.coef_).ravel()[::-1]
    vocab_list = get_vocab_list()[:, 1]
    for i in range(15):
        print(' %-15s (%f) \n' % (vocab_list[index_array[i]], Classification.coef_[:, index_array[i]]))

    print('Program paused. Press enter to continue.\n')
    # pause_func()

    # =================== Part 6: Try Your Own Emails =====================
    filename = 'emailSample1.txt'
    predict_email_spam(filename)

    filename = 'spamSample1.txt'
    predict_email_spam(filename)

    filename = 'emailSample2.txt'
    predict_email_spam(filename)

    filename = 'spamSample2.txt'
    predict_email_spam(filename)
