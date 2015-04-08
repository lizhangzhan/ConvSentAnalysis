import lasagne
import theano
import theano.tensor as T
import time

import utils

from mnist import create_iter_functions, train

num_epochs = 100
random_state = 1066

def load_data(random_state=1066, n=1000, max_phrase_length=100):
    data = utils.load_data(random_state=random_state,
                           n=n,
                           max_phrase_length=max_phrase_length)

    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    X_train = X_train.reshape((-1, max_phrase_length, 67)).transpose(0, 2, 1)
    X_valid = X_valid.reshape((-1, max_phrase_length, 67)).transpose(0, 2, 1)
    X_test = X_test.reshape((-1, max_phrase_length, 67)).transpose(0, 2, 1)

    # Robert: what about reshaping this data for 1D convs?
    # hstack() instead of hstack() in when creatign X in utils?

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        #input_height=X_train.shape[2], # what's the equivalent in our vectors?
        #input_width=X_train.shape[3],
        output_dim=5, # since five sentiment class
        )

def build_model(batch_size, max_phrase_length, output_dim):
    l_in = lasagne.layers.InputLayer((batch_size, 67, max_phrase_length))

    l_conv1 = lasagne.layers.Conv1DLayer(l_in, filter_length=5, border_mode='same', num_filters=128)
    l_pool1 = lasagne.layers.MaxPool1DLayer(l_conv1, ds=3)

    l_conv2 = lasagne.layers.Conv1DLayer(l_pool1, filter_length=5, border_mode='same', num_filters=128)
    l_pool2 = lasagne.layers.MaxPool1DLayer(l_conv2, ds=3)

    l_conv3 = lasagne.layers.Conv1DLayer(l_pool2, filter_length=3, border_mode='same', num_filters=128)
    l_pool3 = lasagne.layers.MaxPool1DLayer(l_conv3, ds=3)

    l_hidden1 = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_pool3, p=0.5), num_units=256)

    l_out = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_hidden1, p=0.5),
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def main():
    BATCH_SIZE = 100
    MAX_PHRASE_LENGTH = 108
    NUM_DATAPOINTS = 10000

    print("Loading data...")
    dataset = load_data(n=NUM_DATAPOINTS, max_phrase_length=MAX_PHRASE_LENGTH)

    print("Building model and compiling functions...")
    output_layer = build_model(
        #input_height=dataset['input_height'],
        #input_width=dataset['input_width'],
        batch_size=BATCH_SIZE,
        max_phrase_length=MAX_PHRASE_LENGTH,
        output_dim=dataset['output_dim'],
    )

    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        X_tensor_type=T.tensor3,
        batch_size=BATCH_SIZE,
        learning_rate=0.001,
        momentum=0.9,
        )

    print("Starting training...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset, BATCH_SIZE):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass

    return output_layer


if __name__ == '__main__':
    main()







