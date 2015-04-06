from utils import *
from sklearn.cross_validation import train_test_split

random_state = 1066

# get the data set
data = get_data(n=1000)

# print some statistics
print_input_statistics(data)

vocab_size = len(get_one_hot_vectors().keys())
print "vocab size: "+str(vocab_size)

print data[100][0]

X, y = vectorize(data, 100)
print X.shape

# get train-test split:
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=0.10,
                                        random_state=random_state)
# get train-test split:
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train,
                                        test_size=0.10,
                                        random_state=random_state)

print X_train.shape
print X_dev.shape
print X_test.shape

