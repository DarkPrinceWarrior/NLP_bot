import json

import nltk_utils
import numpy as np

def jsonOpen():
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    return intents


if __name__ == "__main__":

    all_words = []
    tags = []
    results = []

    intents = jsonOpen()
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = nltk_utils.tokenize(pattern)
            all_words.extend(w)
            results.append((w,tag))

    ignore_words = ['.', '?', '!', ',']

    all_words = [nltk_utils.stemming(w) for w in all_words
                 if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(tags)

    x_train = []
    y_train = []

    for (pattern_sentence, tag) in results:
        # X: bag of words for each pattern_sentence
        bag = nltk_utils.bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)


