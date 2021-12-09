import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import models
from keras.applications.densenet import layers


def model_create(input_size,hidden_size,output_size):
    model = models.Sequential()
    model.add(layers.Dense(hidden_size, activation="relu", input_shape=(input_size,)))
    model.add(layers.Dense(hidden_size, activation="relu"))
    model.add(layers.Dense(output_size, activation="softmax"))

    return model

def model_compile(model, loss_function, metrics):
    model.compile(optimizer='rmsprop',
                  loss=loss_function,
                  metrics=[metrics])

    return model


def model_fit(model,partial_x_train,partial_y_train,
              x_val,y_val,
              epochs,
              batch_size):
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val))

    return history


def model_evaluation():
    pass
