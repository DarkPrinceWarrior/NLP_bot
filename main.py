import json


import nltk_utils
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
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


    model = models.Sequential()
    model.add(layers.Dense(8,activation="relu", input_shape=(len(all_words),)))
    model.add(layers.Dense(8,activation="relu"))
    model.add(layers.Dense(7,activation="softmax"))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = x_train[:5]
    partial_x_train = x_train[5:]

    y_val = y_train[:5]
    partial_y_train = y_train[5:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=4,
                        batch_size=8,
                        validation_data=(x_val,y_val))

    print(history)


