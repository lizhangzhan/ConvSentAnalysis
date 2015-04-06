from utils import *

# get the data set
data = get_data(n=1000)

# print some statistics
print_input_statistics(data)

vocab_size = len(get_one_hot_vectors().keys())
print "vocab size: "+str(vocab_size)

print data[100][0]

X, y = vectorize(data, 100)
print X.shape

# each char vector has length=267; we have 66 chars, so each phrase vector has length=267 * 67 = 17.889