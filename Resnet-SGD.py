import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, AveragePooling2D, Input
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Model
from keras.regularizers import l2
import keras

#from matplotlib import pyplot as plt
print_var= []
print_var1= []

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

# Selects correct version of ResNet
n = 18
version = 1

# Defines a single ResNet layer (from Keras example codes)
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
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
def define_model(input_shape=(32,32,3), depth=110, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
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

with tf.device('/gpu:0'):    # Ensures running on GPU
	model = define_model()
    
    learning_rate = 0.1
    num_examples = len(x_train)
    batch_size = 100
    num_epochs = 250
    
    history = LossHistory()
    sgd=SGD(lr=learning_rate, momentum=0.9)   # Set optimization method to SGD
    
    for epoch in range(num_epochs):
        x_train, y_train = shuffle(x_train, y_train) # shuffle training data
        datagen = ImageDataGenerator(width_shift_range=4, height_shift_range=4, horizontal_flip=True) # data augmentation
        x_train_aug, y_train_aug = next(datagen.flow(x_train, y_train, batch_size=num_examples)) 
        
        if epoch == 150 or epoch == 220: # learning rate scheduler as defined in paper we are reproducing
            learning_rate *= 1/10
            sgd=SGD(lr=learning_rate, momentum=0.9)
        
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # Compile the model
        
        history.clear_history()
        model.fit(x=x_train_aug, y=y_train_aug, batch_size=batch_size, callbacks=[history], shuffle=False) # Train the model
        
        # Gradient calculation and recording - one file records every batch and the other saves at the end of each epoch
		grads = history.losses
        for i in range(1,len(grads)):
            mean = sum(grads[:i]) / len(grads[:i])
            var = sum([(grad - mean)**2 for grad in grads[:i]]) / len(grads[:i])
            if (epoch%50==0):
                f1 = open('ressdgf1.txt','w')
                f1.write('\n' + str(var))
                f1.close()
            if (i==(len(grads)-1)):
                f12 = open('ressdgf2.txt','w')
                f2.write('\n' + str(var))
                f2.close()
    
        model.evaluate(x=x_test, y=y_test, batch_size=100)


#f1.close()
#f2.close()
#plt.figure(1)
#plt.plot(print_var)
#plt.ylabel('Variance')
#
#
#plt.figure(2)
#plt.plot(print_var1)
#plt.ylabel('Variance')
#plt.show()