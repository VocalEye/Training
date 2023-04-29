# Based on "GoogleNet Architecture Implementation in Keras with CIFAR-10 Dataset"
# Avaialble at
# https://machinelearningknowledge.ai/googlenet-architecture-implementation-in-keras-with-cifar-10-dataset/

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input, concatenate
from keras.models import Model
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import os
from datetime import datetime
from src.utils.writeReport import writeRecordOnReport

class MiniGoogleNet:
    model = Model
    callbacks = []
    name = "miniGoogLeNet"
    runName = name
    
    inputShape = ()
    batchSize = 64
    epochs = 100
    learningRate = 5e-3
    momentum = 0.9
    augmentations = {}

    def __init__(self, inputShape, classes, batchSize, epochs, learningRate, momentum, augmentations, logsOutput):
        chanDim = -1

        self.inputShape = inputShape
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.momentum = momentum
        self.augmentations = ImageDataGenerator(**augmentations)

        # (Step 1) Define the model input
        inputs = Input(shape = self.inputShape)

        # First CONV module
        x = self.convolutionModule(inputs, 96, 3, 3, (1, 1), chanDim)

        # (Step 2) Two Inception modules followed by a downsample module
        x = self.inceptionModule(x, 32, 32, 32, 32, chanDim)
        x = self.inceptionModule(x, 32, 48, 48, 32, chanDim)
        x = self.downsampleModule(x, 80, chanDim)

        # (Step 3) Five Inception modules followed by a downsample module
        x = self.inceptionModule(x, 112, 48, 32, 48, chanDim)
        x = self.inceptionModule(x, 96, 64, 32, 32, chanDim)
        x = self.inceptionModule(x, 80, 80, 32, 32, chanDim)
        x = self.inceptionModule(x, 48, 96, 32, 32, chanDim)
        x = self.inceptionModule(x, 112, 48, 32, 48, chanDim)
        x = self.downsampleModule(x, 96, chanDim)

        # (Step 4) Two Inception modules followed
        x = self.inceptionModule(x, 176, 160, 96, 96, chanDim)
        x = self.inceptionModule(x, 176, 160, 96, 96, chanDim)

        # Global POOL and dropout
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # (Step 5) Softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # Create the model
        self.model = Model(inputs, x, name = self.name)

        self.runName += "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callbacks = [
            LearningRateScheduler(self.polynomialDecay),
            # EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5),
            TensorBoard(log_dir=logsOutput + self.runName, histogram_freq=1)
        ]
    
    def compileModel(self):
        self.model.compile(
            loss = "categorical_crossentropy",
            optimizer = SGD(
                learning_rate = self.learningRate, 
                momentum = self.momentum,
                clipnorm=1.0,
                clipvalue=0.5
            ),
            metrics = ["accuracy"])
    
    def fit(self, trainData, testData):
        return self.model.fit(
            self.augmentations.flow(trainData[0], trainData[1], batch_size = self.batchSize),
            validation_data = testData,
            steps_per_epoch = len(trainData[0]) // self.batchSize,
            epochs = self.epochs, 
            callbacks = self.callbacks, 
            verbose = 1)

    def save(self, output):
        if not os.path.exists(output):
            os.makedirs(output)
        self.model.save(output + self.runName + '.h5')

    def polynomialDecay(self, epoch):
        maxEpochs = self.epochs
        baseLR = self.learningRate
        power = 1.0
        alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
        return alpha

    def convolutionModule(self, input, No_of_filters, filtersizeX, filtersizeY, stride, chanDim, padding = "same"):
        input = Conv2D(
            No_of_filters,
            (filtersizeX, filtersizeY),
            strides = stride,
            padding = padding)(input)
        input = BatchNormalization(axis = chanDim)(input)
        input = Activation("relu")(input)
        return input

    def inceptionModule(self, input,numK1x1,numK3x3,numk5x5,numPoolProj,chanDim):
        #Step 1
        conv_1x1 = self.convolutionModule(input, numK1x1, 1, 1,(1, 1), chanDim) 
        #Step 2
        conv_3x3 = self.convolutionModule(input, numK3x3, 3, 3,(1, 1), chanDim)
        conv_5x5 = self.convolutionModule(input, numk5x5, 5, 5,(1, 1), chanDim)
        #Step 3
        pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
        pool_proj = Conv2D(numPoolProj, (1, 1), padding='same', activation='relu')(pool_proj)
        #Step 4
        input = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=chanDim)
        return input

    def downsampleModule(self, input, No_of_filters, chanDim):
        conv_3x3 = self.convolutionModule(
            input,
            No_of_filters,
            3,
            3,
            (2, 2),
            chanDim,
            padding = "valid"
        )
        pool = MaxPooling2D((3,3),strides=(2,2))(input)
        input = concatenate([conv_3x3,pool], axis=chanDim)
        return input

    def getName(self):
        return self.runName
    
    def getSpecialValues(self, configuration):
        return {
            "momentum": self.momentum,
            "hard-test": configuration["hardTest"] if "hardTest" in configuration else False
        }
