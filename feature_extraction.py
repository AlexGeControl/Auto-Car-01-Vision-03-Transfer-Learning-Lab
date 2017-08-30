import pickle

import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import Adam

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string(
    'training_file',
    'features/GoogLeNet/inception_cifar10_100_bottleneck_features_train.p',
    "Bottleneck features training file (.p)"
)
flags.DEFINE_string(
    'validation_file',
    'features/GoogLeNet/inception_cifar10_bottleneck_features_validation.p',
    "Bottleneck features validation file (.p)"
)

flags.DEFINE_integer(
    'epochs',
    50,
    "The number of epochs."
)
flags.DEFINE_integer(
    'batch_size',
    256,
    "The batch size."
)

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_val, y_val, X_train, y_train = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # Define the model
    N_CLASSES = len(np.unique(y_train))
    INPUT_SHAPE = X_train.shape[1: ]

    '''
    # Style 01--Sequential
    model = Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(Dense(N_CLASSES, activation='softmax'))
    model.summary()
    '''
    # Style 02--Functional
    inp = Input(shape=INPUT_SHAPE)
    x = Flatten()(inp)
    x = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(inp, x)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    # Train your model here
    history = model.fit(
        X_train, y_train,
        batch_size = FLAGS.batch_size, nb_epoch = FLAGS.epochs,
        verbose=1,
        validation_data=(X_val, y_val)
    )

    score = model.evaluate(X_val, y_val, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
