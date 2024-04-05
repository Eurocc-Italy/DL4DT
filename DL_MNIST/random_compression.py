import numpy as np

m=784
N=60000

s=50
n=32173

c = (m*n)+(s*N)
compression = (1-(c/(m*N)))*100
chosen_columns_number = int((N/100)*(100-compression))  #6000
chosen_col_num_test = int(chosen_columns_number/6)

cols = np.random.randint(0, N, size=chosen_columns_number, dtype=int)
cols_test = np.random.randint(0, 10000, size=chosen_col_num_test, dtype=int)

# X train
X_train = np.load('../dataset_MNIST/x_train.npy')
X_train = X_train/255
X=np.transpose(np.reshape(X_train,[X_train.shape[0],784]))

X=X[:,cols]

X = np.reshape(np.transpose(X),[chosen_columns_number,28,28])
np.save('../dataset_MNIST/mnist_compressed_random.npy',X)

# train labels

mnist_labels = np.load('../dataset_MNIST/y_train.npy')

mnist_labels = mnist_labels[cols]

np.save('../dataset_MNIST/mnist_compressed_random_labels.npy',mnist_labels)

print(X.shape)
print(mnist_labels.shape)


# X test

X_test = np.load('../dataset_MNIST/x_test.npy')
X_test = X_test/255
X = np.transpose(np.reshape(X_test,[X_test.shape[0],784]))

X = X[:,cols_test]

X = np.reshape(np.transpose(X),[chosen_col_num_test,28,28])
np.save('../dataset_MNIST/mnist_compressed_random_test.npy',X)

# test labels

mnist_labels = np.load('../dataset_MNIST/y_test.npy')

mnist_labels = mnist_labels[cols_test]

np.save('../dataset_MNIST/mnist_compressed_random_labels_test.npy',mnist_labels)

print(X.shape)
print(mnist_labels.shape)
