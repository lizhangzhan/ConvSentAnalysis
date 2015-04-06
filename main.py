import lasagne

import utils

random_state = 1066

def load_data(random_state=1066, n=1000, max_phrase_length=100):
	data = utils.load_data(random_state=random_state,
		                   n=n,
		                   max_phrase_length=max_phrase_length)

	X_train, y_train = data[0]
	X_valid, y_valid = data[1]
	X_test, y_test = data[2]

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
        input_height=X_train.shape[2],
        input_width=X_train.shape[3],
        output_dim=10,
        )





