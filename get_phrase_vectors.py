import os
import numpy
from collections import defaultdict


def get_data():
    """Returns a list of phrase, sentiment class tuples. Phrases range from 3 to 267 characters.
    Classes range from 1 (very negative) to 5 (very positive)."""

    dir_ = os.getcwd() + '/stanfordSentimentTreebank/'

    phrase_dict = dict()

    # store phrases by index
    with open(dir_ + 'dictionary.txt', 'r') as phrase_file:
        for phrase_idx in phrase_file:
            phrase, idx = phrase_idx.strip().split('|')
            phrase = phrase.lower()
            phrase_dict[idx] = phrase

    # get the sentiment class for each phrase and return them as a list of phrase-class tuples
    data = []
    with open(dir_ + 'sentiment_labels.txt', 'r') as sentiment_file:
        sentiment_file.next()
        for idx_sentiment in sentiment_file:
            idx, sentiment = idx_sentiment.strip().split('|')
            class_ = sentiment_to_class(sentiment)
            data.append((phrase_dict[idx], class_))

    return data


def sentiment_to_class(sentiment):
    """
    Convert sentiment probability to one of five classes, according to the following cut-offs
    (see './stanfordSentimentTreebank/README.txt'):

    [0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]
    """
    sentiment = float(sentiment)

    ret = None

    if sentiment <= 0.2:
        ret = 1
    elif sentiment <= 0.4:
        ret = 2
    elif sentiment <= 0.6:
        ret = 3
    elif sentiment <= 0.8:
        ret = 4
    elif sentiment <= 1.0:
        ret = 5

    return ret


def print_input_statistics(data):

    phrase_lengths = [len(phrase) for phrase, class_ in data]

    print 'Input statistics (in characters)'
    print 'Longest phrase:\t\t%s' % max(phrase_lengths)
    print 'Shortest phrase:\t%s' % min(phrase_lengths)
    print 'Mean phrase length:\t%s (std: %s)' % (numpy.mean(phrase_lengths), numpy.std(phrase_lengths))



def get_one_hot_vectors():
    """ Construct one-hot vectors for the chars below """

    english_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z']

    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    special_chars = ['-', ',', ';', '.', '!', '?', ':', '"', '/', '\\',  '|', '_', '@', '#', '$', '%', '&', '*',
                     '+', '-', '=', '<', '>', '(', ')', '[', ']', '{', '}', '`', "'"]

    # does not cover 2.448 non-ASCII char tokens, out of a total of 10.207.891 char tokens in the data set
    # (> 99.999 % of all char tokens are covered)
    chars = english_chars + digits + special_chars

    # return an all zero vector as the default case
    one_hot_vector_dict = defaultdict(lambda: numpy.zeros(len(chars)))

    for idx, char in enumerate(chars):
        vector = numpy.zeros(len(chars))
        vector[idx] = 1
        one_hot_vector_dict[char] = vector

    return one_hot_vector_dict


def quantize(phrase, one_hot_vector_dict, max_phrase_length):
    """
    Concatenate one-hot vectors for chars in 'string_'.
    Add zero-vectors if 'string_' is shorter than the longest string in the data set.
    """
    # if 'phrase' is shorter than the longest phrase in the data set, add empty strings
    # this ensures equal length of all phrase vectors
    if len(phrase) < max_phrase_length:
        nr_empty_chars = max_phrase_length - len(phrase)
        phrase += ''.join([' ' for i in range(nr_empty_chars)])

    # get one-hot vectors for all chars in the phrase
    vs = [one_hot_vector_dict[char] for char in phrase]

    assert len(vs) == max_phrase_length

    # return concatenated char vectors
    return numpy.hstack(vs)


#######################################################################################################################


# get the data set
data = get_data()

# print the  first 10 data points
for phrase, class_ in data[:10]:
    print class_, phrase
print
print

# print some statistics
print_input_statistics(data)

# get one-hot vector dict and quantize phrases
one_hot_vector_dict = get_one_hot_vectors()
phrase_vectors = [quantize(phrase, one_hot_vector_dict, max_phrase_length=267) for phrase, class_ in data]


# each char vector has length=267; we have 67 chars, so each phrase vector has length=267 * 67 = 17.889