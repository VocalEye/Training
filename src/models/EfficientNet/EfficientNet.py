from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import os
from datetime import datetime
from src.utils.writeReport import writeRecordOnReport

class EfficientNet:
    model = Model
    callbacks = []
    name = "EfficientNet"
    runName = name
    
    inputShape = ()
    batchSize = 64
    epochs = 100
    learningRate = 5e-3
    momentum = 0.9
    augmentations = {}

    def __init__(self, inputShape, classes, batchSize, epochs, learningRate, momentum, augmentations, logsOutput):
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.momentum = momentum
        self.augmentations = ImageDataGenerator(**augmentations)

        # Create the model
        self.model = EfficientNetV2B0(
            include_top = True,
            weights = None,
            input_shape = self.inputShape,
            classes = classes,
            classifier_activation = "softmax"
        )
        
        self.runName += "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.callbacks = [
            LearningRateScheduler(self.polynomialDecay),
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

    def getName(self):
        return self.runName
    
    def getSpecialValues(self, configuration):
        return {
            "momentum": self.momentum,
            "hard-test": configuration["hardTest"] if "hardTest" in configuration else False
        }

    def polynomialDecay(self, epoch):
        maxEpochs = self.epochs
        baseLR = self.learningRate
        power = 1.0
        alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
        return alpha