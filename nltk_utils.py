import nltk

# nltk.download('punkt') already downloaded
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stemming(word):
    return stemmer.stem(word.lower())


def excl_punk(sentence):
    pass

def bag_of_words(token_sentence, all_words):
    pass