###############################################################################
#                                                                             #
#  DL4DT                                                                      #
#                                                                             #
#  L.Cavalli                                                                  #
#                                                                             #
#  Copyright (C) 2024 CINECA HPC department                                   #
#                                                                             #
#  This program is free software; you can redistribute it and/or modify it    #
#  under the terms of the GNU Lesser General Public License as published by   #
#  the Free Software Foundation; either version 3 of the License, or          #
#  (at your option) any later version.                                        #
#                                                                             #
#  This program is distributed in the hope that it will be useful,            #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          #
#  Lesser General Public License for more details.                            #
#                                                                             #
#  You should have received a copy of the GNU Lesser General Public License   #
#  along with this program; if not, write to the Free Software Foundation,    #
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.        #
#                                                                             #
###############################################################################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import keras.backend as K
from sklearn import model_selection
import tensorflow as tf
from sklearn.preprocessing import normalize
import horovod.tensorflow as hvd

# Horovod: initialize Horovod.
hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# load the dataset 

num_classes=10
input_shape=(28,28,1)
x_train = np.load('../dataset_MNIST/mnist_DL_train_n27704_s30.npy') #put the compressed dataset or x_train.npy
y_train = np.load('../dataset_MNIST/y_train.npy')
X_test = np.load('../dataset_MNIST/x_test.npy') #put the X_test rewritten with the dictionary or x_test.npy
Y_test = np.load('../dataset_MNIST/y_test.npy')

print("SHAPE:")
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
print('x_test shape',X_test.shape)
x_test=np.reshape(X_test.T,(X_test.shape[1],28,28))
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("SHAPE:")
print(x_train.shape, "train shape")
print(x_test.shape, "test shape")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

scaled_lr = 0.001*hvd.size()
opt = tf.optimizers.Adam(scaled_lr)
opt = hvd.DistributedOptimizer(opt, sparse_as_dense=True, compression=hvd.Compression.fp16)

# compile and fit
mnist_model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)

callbacks = [
    hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.keras.callbacks.MetricAverageCallback(),
    hvd.keras.callbacks.LearningRateWarmupCallback(warmup_epochs=3, initial_lr=scaled_lr, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
import time
t0=time.time()
batch_size=128
hist = mnist_model.fit(x_train,y_train, batch_size=batch_size, callbacks=callbacks, epochs=24, verbose=verbose)

t1=time.time()
total=t1-t0

print("Total training time: ",total)
print(hist.history)
train_accuracy = hist.history['accuracy']
train_loss = hist.history['loss']

#print("Training loss: ", train_loss)
#print("Training accuracy: ", train_accuracy)

# evaluation

test_loss, test_acc = mnist_model.evaluate(x_test, y_test,verbose=0)

print("Test accuracy", test_acc)
print("Test loss", test_loss)


######## accuracy of classes ###########

y_test=Y_test
x_test=X_test.T

# class 0
idx_zero=np.argwhere(y_test==0)
y_test_0=y_test[np.squeeze(idx_zero)]
x_test_0=x_test[np.squeeze(idx_zero),:]

x_test_0 = np.reshape(x_test_0,(x_test_0.shape[0],28,28))
x_test_0 = np.expand_dims(x_test_0, -1)
y_test_0 = keras.utils.to_categorical(y_test_0, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_0, y_test_0)

print("Test accuracy class 0", test_acc)
print("Test loss class 0", test_loss)


# class 1
idx_one=np.argwhere(y_test==1)
y_test_1=y_test[np.squeeze(idx_one)]
x_test_1=x_test[np.squeeze(idx_one),:]

x_test_1=np.reshape(x_test_1,(x_test_1.shape[0],28,28))
x_test_1 = np.expand_dims(x_test_1, -1)
y_test_1 = keras.utils.to_categorical(y_test_1, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_1, y_test_1)

print("Test accuracy class 1", test_acc)
print("Test loss class 1", test_loss)

# class 2
idx_two=np.argwhere(y_test==2)
y_test_2=y_test[np.squeeze(idx_two)]
x_test_2=x_test[np.squeeze(idx_two),:]

x_test_2=np.reshape(x_test_2,(x_test_2.shape[0],28,28))
x_test_2= np.expand_dims(x_test_2, -1)
y_test_2 = keras.utils.to_categorical(y_test_2, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_2, y_test_2)

print("Test accuracy class 2", test_acc)
print("Test loss class 2", test_loss)

# class 3
idx_three=np.argwhere(y_test==3)
y_test_3=y_test[np.squeeze(idx_three)]
x_test_3=x_test[np.squeeze(idx_three),:]

x_test_3=np.reshape(x_test_3,(x_test_3.shape[0],28,28))
x_test_3 = np.expand_dims(x_test_3, -1)
y_test_3 = keras.utils.to_categorical(y_test_3, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_3, y_test_3)

print("Test accuracy class 3", test_acc)
print("Test loss class 3", test_loss)

# class 4
idx_four=np.argwhere(y_test==4)
y_test_4=y_test[np.squeeze(idx_four)]
x_test_4=x_test[np.squeeze(idx_four),:]

x_test_4 =np.reshape(x_test_4,(x_test_4.shape[0],28,28))
x_test_4 = np.expand_dims(x_test_4, -1)
y_test_4 = keras.utils.to_categorical(y_test_4, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_4, y_test_4)

print("Test accuracy class 4", test_acc)
print("Test loss class 4", test_loss)

# class 5
idx_five=np.argwhere(y_test==5)
y_test_5=y_test[np.squeeze(idx_five)]
x_test_5=x_test[np.squeeze(idx_five),:]

x_test_5=np.reshape(x_test_5,(x_test_5.shape[0],28,28))
x_test_5 = np.expand_dims(x_test_5, -1)
y_test_5 = keras.utils.to_categorical(y_test_5, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_5, y_test_5)

print("Test accuracy class 5", test_acc)
print("Test loss class 5", test_loss)


# class 6
idx_six=np.argwhere(y_test==6)
y_test_6=y_test[np.squeeze(idx_six)]
x_test_6=x_test[np.squeeze(idx_six),:]

x_test_6 =np.reshape(x_test_6,(x_test_6.shape[0],28,28))
x_test_6 = np.expand_dims(x_test_6, -1)
y_test_6 = keras.utils.to_categorical(y_test_6, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_6, y_test_6)

print("Test accuracy class 6", test_acc)
print("Test loss class 6", test_loss)

# class 7
idx_seven=np.argwhere(y_test==7)
y_test_7=y_test[np.squeeze(idx_seven)]
x_test_7=x_test[np.squeeze(idx_seven),:]

x_test_7 =np.reshape(x_test_7,(x_test_7.shape[0],28,28))
x_test_7 = np.expand_dims(x_test_7, -1)
y_test_7 = keras.utils.to_categorical(y_test_7, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_7, y_test_7)

print("Test accuracy class 7", test_acc)
print("Test loss class 7", test_loss)

# class 8
idx_eig=np.argwhere(y_test==8)
y_test_8=y_test[np.squeeze(idx_eig)]
x_test_8=x_test[np.squeeze(idx_eig),:]

x_test_8=np.reshape(x_test_8,(x_test_8.shape[0],28,28))
x_test_8 = np.expand_dims(x_test_8, -1)
y_test_8 = keras.utils.to_categorical(y_test_8, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_8, y_test_8)
print("Test accuracy class 8", test_acc)
print("Test loss class 8", test_loss)

# class 9
idx_nine=np.argwhere(y_test==9)
y_test_9=y_test[np.squeeze(idx_nine)]
x_test_9=x_test[np.squeeze(idx_nine),:]

x_test_9=np.reshape(x_test_9,(x_test_9.shape[0],28,28))
x_test_9 = np.expand_dims(x_test_9, -1)
y_test_9 = keras.utils.to_categorical(y_test_9, num_classes)

test_loss, test_acc = mnist_model.evaluate(x_test_9, y_test_9)

print("Test accuracy class 9", test_acc)
print("Test loss class 9", test_loss)


##############################################

# clean the model 
import keras.backend as K
del mnist_model
K.clear_session()
