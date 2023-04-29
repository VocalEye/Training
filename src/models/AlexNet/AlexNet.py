
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
from datetime import datetime

class AlexNet:
    model = Model
    callbacks = []
    name = "AlexNet"
    runName = name

    inputShape = ()
    batchSize = 64
    epochs = 100
    learningRate = 5e-3
    augmentations = {}

    def __init__(self, inputShape, classes, batchSize, epochs, learningRate, augmentations, logsOutput):
        #Instantiation
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.augmentations = ImageDataGenerator(**augmentations)

        self.model = Sequential()

        #1st Convolutional Layer
        self.model.add(Conv2D(filters = 96, input_shape=self.inputShape, kernel_size=(11,11), strides=(4,4), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #2nd Convolutional Layer
        self.model.add(Conv2D(filters = 256, kernel_size=(5, 5), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #3rd Convolutional Layer
        self.model.add(Conv2D(filters = 384, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        #4th Convolutional Layer
        self.model.add(Conv2D(filters = 384, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        #5th Convolutional Layer
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #Passing it to a Fully Connected layer
        self.model.add(Flatten())
        # 1st Fully Connected Layer
        self.model.add(Dense(4096, input_shape=self.inputShape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        self.model.add(Dropout(0.4))

        #2nd Fully Connected Layer
        self.model.add(Dense(4096))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        #Add Dropout
        self.model.add(Dropout(0.4))

        #3rd Fully Connected Layer
        self.model.add(Dense(1000))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        #Add Dropout
        self.model.add(Dropout(0.4))

        #Output Layer
        self.model.add(Dense(classes))
        self.model.add(BatchNormalization())
        self.model.add(Activation('softmax'))

        self.runName += "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callbacks = [
            ReduceLROnPlateau(monitor='val_accuracy', factor=.01, patience=3, min_lr=1e-5),
            TensorBoard(log_dir=logsOutput + self.runName, histogram_freq=1)
        ]

    def compileModel(self):
        self.model.compile(
            loss = "categorical_crossentropy", 
            optimizer= Adam(
                learning_rate = self.learningRate,
                clipnorm = 1.0,
                clipvalue = 0.5,
            ), 
            metrics=['accuracy']
        )
    
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
    
    def getName(self):
        return self.runName

    def getSpecialValues(self, configuration):
        return {
            "hard-test": configuration["hardTest"] if "hardTest" in configuration else False
        }