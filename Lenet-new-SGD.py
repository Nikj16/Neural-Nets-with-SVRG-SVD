import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
print_var= []
print_var1= []
f1 = open('sgdf1.txt','w')
f2 = open('sgdf2.txt','w')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batches = []
for index in range(1,6):
    batches.append(unpickle('./cifar-10-batches-py/data_batch_{}'.format(index)))
test_batch = (unpickle('./cifar-10-batches-py/test_batch'))

x_train = batches[0][b'data']
y_train = batches[0][b'labels']
for i in range(1,len(batches)):
    x_train = np.concatenate((x_train, batches[i][b'data']), axis=0)
    y_train = y_train + batches[i][b'labels']
    
x_test = test_batch[b'data']
y_test = test_batch[b'labels']

x_train = np.reshape(x_train, [50000,32,32,3])
x_test = np.reshape(x_test, [10000,32,32,3])

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)


print("Input shape: {}".format(x_train[0].shape))
print("Training Set:   {} samples".format(len(x_train)))
print("Test Set:       {} samples".format(len(x_test)))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def clear_history(self, logs={}):
        self.losses = []



def define_model(training):
    model = Sequential()

    model.add(Conv2D(6, (5,5), activation='relu', input_shape=(32,32,3), trainable=training))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=False, scale=False, name='BN1'))
    model.add(Conv2D(16, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), trainable=training))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=False, scale=False, name='BN2'))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', trainable=training))
    model.add(Dense(84, activation='relu', trainable=training))
    model.add(Dense(10, activation='softmax', trainable=training))

    return model



model = define_model(True)

learning_rate = 0.1
num_examples = len(x_train)
batch_size = 100
num_epochs = 250

history = LossHistory()
sgd=SGD(lr=learning_rate, momentum=0.9)


for epoch in range(num_epochs):
    x_train, y_train = shuffle(x_train, y_train)
    datagen = ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=True)
    x_train_aug, y_train_aug = next(datagen.flow(x_train, y_train, batch_size=num_examples))
    
    if epoch == 150 or epoch == 220:
        learning_rate *= 1/10
        sgd=SGD(lr=learning_rate, momentum=0.9)
    
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    history.clear_history()
    model.fit(x=x_train_aug, y=y_train_aug, batch_size=batch_size, callbacks=[history], shuffle=False)
    
    grads = history.losses
    batch_vars = []

    for i in range(1,len(grads)):
        mean = sum(grads[:i]) / len(grads[:i])
        var = sum([(grad - mean)**2 for grad in grads[:i]]) / len(grads[:i])
        #print(var)
        print_var.append(var)
        f1.write('\n' + str(var))
        if (i==(len(grads)-1)):
        	print_var1.append(var)
        	f2.write('\n' + str(var))
        batch_vars.append(var)

    model.evaluate(x=x_test, y=y_test, batch_size=100)

f1.close()
f2.close()
plt.figure(1)
plt.plot(print_var)
plt.ylabel('Variance')


plt.figure(2)
plt.plot(print_var1)
plt.ylabel('Variance')
plt.show()
