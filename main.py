import json


import nltk_utils
import numpy as np
import os

from model import model_create, model_compile, model_fit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils.np_utils import to_categorical

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
            results.append((w, tag))

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
    # y_train = np.array(y_train)

    y_train = to_categorical(y_train, 7)

    # create the NN model
    input_size = len(all_words)
    # number of hidden neurons
    hidden_size = 8
    # number of output neurons
    output_size = 7
    model = model_create(input_size, hidden_size, output_size)

    # compile the model
    loss = "categorical_crossentropy"
    metrics = "accuracy"
    model = model_compile(model, loss, metrics)


    # fit the model and get the results
    x_val = x_train[:5]
    partial_x_train = x_train[5:]

    y_val = y_train[:5]
    partial_y_train = y_train[5:]
    epochs = 2
    batch_size = 8
    history = model_fit(model,partial_x_train,partial_y_train,
                        x_val,y_val,
                        epochs,batch_size)



    print(history)


