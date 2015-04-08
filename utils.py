import os
import codecs
import re
import numpy as np

from sklearn.cross_validation import train_test_split

WHITESPACE = re.compile(r"\s+")

def parse_data(n=1000):
    """Returns a list of phrase, sentiment class tuples. Phrases range from 3 to 267 characters.
    Classes range from 1 (very negative) to 5 (very positive)."""

    dir_ = os.getcwd() + '/stanfordSentimentTreebank/'
    # get evaluations as np array:
    evaluations = np.zeros(239231+1, dtype=np.int8) # total nb of evaluations + 1
    for line in codecs.open(dir_ + 'sentiment_labels.txt', 'r'):
        idx, sentiment = line.strip().split('|', 1)
        try:
            evaluations[int(idx)] = sentiment_to_class(sentiment)
        except ValueError: # ignore header in 1st line
            pass

    data = []
    cnt = 0
    for line in open(dir_ + 'dictionary.txt', 'r'):
        phrase, idx = line.strip().lower().split('|')
        line = re.sub(WHITESPACE, " ", line)
        data.append((phrase, evaluations[int(idx)]))
        cnt += 1
        if cnt >= n:
            break

    return tuple(data)


def sentiment_to_class(sentiment):
    """
    Convert sentiment probability to one of five classes, according to the following cut-offs
    (see './stanfordSentimentTreebank/README.txt'):

    [0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]
    """
    sentiment = float(sentiment)

    if sentiment >= 0.8:
        return 4
    elif sentiment >= 0.6:
        return 3
    elif sentiment >= 0.4:
        return 2
    elif sentiment >= 0.2:
        return 1
    else:
        return 0

def print_input_statistics(data):

    phrase_lengths = [len(phrase) for phrase, class_ in data]

    print 'Input statistics (in characters)'
    print 'Longest phrase:\t\t%s' % max(phrase_lengths)
    print 'Shortest phrase:\t%s' % min(phrase_lengths)
    print 'Mean phrase length:\t%s (std: %s)' % (np.mean(phrase_lengths), np.std(phrase_lengths))

def get_one_hot_vectors():
    """ Construct one-hot vectors for the chars below """

    english_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z', ' ']

    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    special_chars = ['-', ',', ';', '.', '!', '?', ':', '"', '/', '\\',  '|', '_', '@', '#', '$', '%', '&', '*',
                     '+', '-', '=', '<', '>', '(', ')', '[', ']', '{', '}', '`', "'"]

    # does not cover 2.448 non-ASCII char tokens, out of a total of 10.207.891 char tokens in the data set
    # (> 99.999 % of all char tokens are covered)
    chars = tuple(set(english_chars + digits + special_chars))

    # return an all zero vector as the default case
    one_hot_vector_dict = {}
    for idx, char in enumerate(chars):
        vector = np.zeros(len(chars), dtype=np.int8)
        vector[idx] = 1
        one_hot_vector_dict[char] = vector

    return one_hot_vector_dict

def pad_or_cut(phrase, max_phrase_length):
    while len(phrase) < max_phrase_length:
        phrase+=" "
    return phrase[:max_phrase_length]


def vectorize(data, max_phrase_length):
    """
    Concatenate one-hot vectors for chars in 'string_'.
    Add zero-vectors if 'string_' is shorter than the longest string in the data set.
    """

    X, y = [], []
    one_hot_vector_dict = get_one_hot_vectors()
    filler = np.zeros(len(one_hot_vector_dict), dtype=np.int8)

    for phrase, class_ in data:
        phrase = pad_or_cut(phrase, max_phrase_length)
        # get one-hot vectors for all chars in the phrase
        vs = []
        for char in phrase:
            try:
                vs.append(one_hot_vector_dict[char])
            except KeyError:
                vs.append(filler)
        assert len(vs) == max_phrase_length
        X.append(np.hstack(vs))
        y.append(class_)

    X = np.asarray(X, dtype=np.int8)
    y = np.asarray(y, dtype=np.int8)

    return X, y


def load_data(random_state, n=10000, max_phrase_length=100):
    # get the data set
    data = parse_data(n)

    # print some statistics
    print_input_statistics(data)

    vocab_size = len(get_one_hot_vectors().keys())
    print "vocab size: "+str(vocab_size)

    X, y = vectorize(data, max_phrase_length)
    print X.shape

    # get train-test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=0.10,
                                            random_state=random_state)
    # get train-test split:
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                            test_size=0.10,
                                            random_state=random_state)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

