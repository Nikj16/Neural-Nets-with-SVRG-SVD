import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, Callback
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, AveragePooling2D, Input
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Model
from keras.regularizers import l2
import keras

from matplotlib import pyplot as plt
print_var= []
print_var1= []
f1 = open('ressvrgf1.txt','w')
f2 = open('ressvrgf2.txt','w')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Loading the CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert string category to binary vector
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

print("Input shape: {}".format(x_train[0].shape))
print("Training Set:   {} samples".format(len(x_train)))
print("Test Set:       {} samples".format(len(x_test)))

# Define a function to call while training each epoch which records the loss
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def clear_history(self, logs={}):
        self.losses = []

class SVRG():
    def __init__(self, whole, batch):
        self.counter = 0
        self.snap_loss_whole = whole
        self.snap_loss_batch = batch
    
    def SVRG_loss(self, y_true, y_pred):
        cross = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
        cross += self.snap_loss_whole + self.snap_loss_batch[self.counter]
        self.counter += 1
        return cross


# Defines a single ResNet layer (from Keras example codes)
def resnet_layer(inputs,
				 trainable,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
	              trainable=trainable,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

# Defines full model by calling ResNet layers and stacking (from Keras example codes)
def define_model(training, input_shape=(32,32,3), depth=110, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, trainable=training)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
			                 trainable=training,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
			                 trainable=training,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
				                 trainable=training,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

model_trainable = define_model(True)
model_frozen = define_model(False)

learning_rate = 0.1
num_examples = len(x_train)
batch_size = 100
num_epochs = 250

history = LossHistory()
sgd=SGD(lr=learning_rate, momentum=0.9)  # Set optimization method to SGD

for epoch in range(num_epochs):
    x_train, y_train = shuffle(x_train, y_train)  # shuffle training data
    datagen = ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=True)  # data augmentation
    x_train_aug, y_train_aug = next(datagen.flow(x_train, y_train, batch_size=num_examples))
    
    if epoch == 150 or epoch == 220:  # learning rate scheduler as defined in paper we are reproducing
        learning_rate *= 1/10
        sgd=SGD(lr=learning_rate, momentum=0.9)
		
		model_frozen.set_weights(model_trainable.get_weights())
    
    model_frozen.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    history.clear_history()
    model_frozen.fit(x=x_train_aug, y=y_train_aug, batch_size=batch_size, callbacks=[history], shuffle=False)
    
    snap_loss_batch = history.losses
    snap_loss_whole = sum(snap_loss_batch) / len(snap_loss_batch)
    
    grads = []
    batch_vars = []
    
    svrg_obj = SVRG(snap_loss_whole, snap_loss_batch)
    
    model_trainable.compile(loss=svrg_obj.SVRG_loss, optimizer=sgd)
    
    history.clear_history()
    model_trainable.fit(x=x_train_aug, y=y_train_aug, batch_size=batch_size, callbacks=[history], shuffle=False)

    grads = history.losses
    batch_vars = []

    # Gradient calculation and recording - one file records every batch and the other saves at the end of each epoch
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


    model_trainable.evaluate(x=x_test, y=y_test, batch_size=100)

f1.close()
f2.close()
plt.figure(1)
plt.plot(print_var)
plt.ylabel('Variance')


plt.figure(2)
plt.plot(print_var1)
plt.ylabel('Variance')
plt.show()