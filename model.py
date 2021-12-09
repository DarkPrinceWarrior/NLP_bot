import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras import models
from keras.applications.densenet import layers
from keras.layers import Dropout
from keras.optimizer_v2.gradient_descent import SGD



def model_create(input_size, hidden_size, output_size):
    model = models.Sequential()
    model.add(layers.Dense(hidden_size, activation="relu", input_shape=(input_size,)))
    model.add(Dropout(0.5))
    model.add(layers.Dense(hidden_size-64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(layers.Dense(output_size, activation="softmax"))

    return model


def model_compile(model, loss_function, metrics):

    sgd = SGD(learning_rate=0.01,decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss=loss_function,
                  metrics=[metrics])

    return model


def model_fit(model,epochs,batch_size, partial_x_train=None, partial_y_train=None,
              x_val=None, y_val=None):

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),shuffle=True, verbose=1)

    return history


def K_fold_validation(x_train, y_train,
                      input_size, hidden_size, output_size,
                      loss_function, metrics, epochs, batch_size):
    fold_number = 4

    num_samples_fold = len(x_train) // fold_number
    all_acc = []
    for i in range(fold_number):
        val_data = x_train[i * num_samples_fold: (i + 1) * num_samples_fold]
        val_targets = y_train[i * num_samples_fold: (i + 1) * num_samples_fold]

        partial_train_data = np.concatenate([x_train[:i * num_samples_fold],
                                             x_train[(i + 1) * num_samples_fold:]],
                                            axis=0)

        partial_train_targets = np.concatenate([y_train[:i * num_samples_fold],
                                                y_train[(i + 1) * num_samples_fold:]],
                                               axis=0)

        model = model_create(input_size, hidden_size, output_size)
        model = model_compile(model, loss_function, metrics)
        history = model_fit(model, epochs,batch_size, partial_train_data, partial_train_targets,
                            val_data, val_targets)

        all_acc.append(history.history["val_accuracy"])

    print(np.mean(all_acc))




def model_evaluation():
    pass
