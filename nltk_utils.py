import nltk

# nltk.download('punkt') already downloaded
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stemming(word):
    return stemmer.stem(word.lower())


def bag_of_words(token_sentence, all_words):
    """
       return bag of words array:
       1 for each known word that exists in the sentence, 0 otherwise
       example:
       sentence = ["hello", "how", "are", "you"]
       words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
       bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
       """

    # stem each word
    sentence_words = [stemming(word) for word in token_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
