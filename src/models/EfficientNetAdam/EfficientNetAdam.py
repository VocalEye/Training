from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import os
from datetime import datetime
from src.utils.writeReport import writeRecordOnReport

class EfficientNetAdam:
    model = Model
    callbacks = []
    name = "EfficientNetAdam"
    runName = name
    
    inputShape = ()
    batchSize = 64
    epochs = 100
    learningRate = 5e-3
    augmentations = {}

    def __init__(self, inputShape, classes, batchSize, epochs, learningRate, augmentations, logsOutput):
        self.inputShape = inputShape
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
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
            "hard-test": configuration["hardTest"] if "hardTest" in configuration else False
        }

    def polynomialDecay(self, epoch):
        maxEpochs = self.epochs
        baseLR = self.learningRate
        power = 1.0
        alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
        return alpha