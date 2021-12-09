import json

import random
import nltk_utils
import numpy as np
import os

from model import model_create, model_compile, model_fit, K_fold_validation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils.np_utils import to_categorical


def jsonOpen():
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    return intents


def build_model(tags, all_words, input_size, hidden_size, output_size,
                loss, metrics, epochs,
                batch_size):

    # create the model
    model = model_create(input_size, hidden_size, output_size)

    # compile the model
    model = model_compile(model, loss, metrics)

    # fit the model and get the results

    x_val = x_train[:3]
    partial_x_train = x_train[3:]

    y_val = y_train[:3]
    partial_y_train = y_train[3:]

    model.fit(x_train,
              y_train,
              epochs=epochs,
              batch_size=batch_size, shuffle=True, verbose=1)


    # model.fit(partial_x_train,
    #           partial_y_train,
    #           epochs=epochs,
    #           batch_size=batch_size,
    #           validation_data=(x_val, y_val), shuffle=True, verbose=1)

    sentence = nltk_utils.tokenize("Which items do you have?")
    sentence = nltk_utils.bag_of_words(sentence, all_words)
    sentence = np.array(sentence)
    sentence = sentence.reshape(1, sentence.shape[0])

    prediction = model.predict(sentence)
    tag_index = np.argmax(prediction)
    print(tag_index)
    print(tags[tag_index])


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

    random.shuffle(results)

    for (pattern_sentence, tag) in results:
        # X: bag of words for each pattern_sentence
        bag = nltk_utils.bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    # print(x_train)
    y_train = to_categorical(y_train, 7)

    # create the NN model
    input_size = len(all_words)
    # number of hidden neurons
    hidden_size = 128
    # number of output neurons
    output_size = 7
    loss = "categorical_crossentropy"
    metrics = "accuracy"
    epochs = 200
    batch_size = 5

    # simple build method
    build_model(tags, all_words, input_size, hidden_size, output_size, loss, metrics, epochs, batch_size)

    # cross-validation method
    # K_fold_validation(x_train, y_train,
    #                   input_size, hidden_size, output_size,
    #                   loss, metrics, epochs, batch_size)
