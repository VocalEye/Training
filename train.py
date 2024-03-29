
from src.models.InceptionV3.InceptionV3 import Inception
from src.models.EfficientNetAdam.EfficientNetAdam import EfficientNetAdam
from src.models.EfficientNet.EfficientNet import EfficientNet
from src.models.ResNetAdam.ResNetAdam import ResNetAdam
from src.models.ResNet.ResNet import ResNet
from src.models.AlexNet.AlexNet import AlexNet
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils.loadDataset import loadDataset
from src.utils.writeReport import writeRecordOnReport
from test import testModel
import gc
import time
import datetime
import os
from numpy import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras import backend as K

np.random.seed(1000)

DEFAULT_TEST = [
    {
        "dataset": './input/test/Dataset 2 - Test/',
        "classSample": None
    }
]

DEFAULT_VALUES = {
    "imageSize": (75, 75, 1),
    "batchSize": 128,
    "epochSize": 100,
    "learningRate": 5e-3,
    "augmentation": {
        "rotation_range": 2,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "fill_mode": "nearest"
    },
    "tests": DEFAULT_TEST
}


NUMBER_PROCESSES = 20
FILENAME = './input/results.csv'
CLASS_SAMPLE = None
INPUTS = {}

def loadInputs():
    global INPUTS
    cases = []
    imageSize = (75, 75, 1)

    copied = DEFAULT_VALUES.copy()
    copied["learningRate"] = 0.040709624769089126
    copied["batchSize"] = 128
    copied["imageSize"] = imageSize
    copied["model"] = "Inception"
    cases.append(copied)

    INPUTS["./input/train/Dataset 2/"] = cases

OUTPUT = './output/models/'
TEST_INPUT = '' # './input/test/Real Test/'
LOGS_OUTPUT = "./output/logs/fit/"

def getModelInstance(configuration):
    if(configuration["model"] == "AlexNet"):
        return AlexNet(
                inputShape = configuration["imageSize"], 
                classes = 9, 
                batchSize = configuration["batchSize"],
                epochs = configuration["epochSize"],
                learningRate = configuration["learningRate"],
                augmentations = configuration["augmentation"],
                logsOutput = LOGS_OUTPUT
            )
    if(configuration["model"] == "Inception"):
        return Inception(
                inputShape = configuration["imageSize"], 
                classes = 9, 
                batchSize = configuration["batchSize"],
                epochs = configuration["epochSize"],
                learningRate = configuration["learningRate"],
                augmentations = configuration["augmentation"],
                momentum = 0.9,
                logsOutput = LOGS_OUTPUT
            )
    elif(configuration["model"] == "ResNet"):
        return ResNet(
                inputShape = configuration["imageSize"], 
                classes = 9, 
                batchSize = configuration["batchSize"],
                epochs = configuration["epochSize"],
                learningRate = configuration["learningRate"],
                augmentations = configuration["augmentation"],
                momentum = 0.9,
                logsOutput = LOGS_OUTPUT
            )
    elif(configuration["model"] == "ResNetAdam"):
        return ResNetAdam(
                inputShape = configuration["imageSize"], 
                classes = 9, 
                batchSize = configuration["batchSize"],
                epochs = configuration["epochSize"],
                learningRate = configuration["learningRate"],
                augmentations = configuration["augmentation"],
                logsOutput = LOGS_OUTPUT
            )
    elif(configuration["model"] == "EfficientNet"):
        return EfficientNet(
                inputShape = configuration["imageSize"], 
                classes = 9, 
                batchSize = configuration["batchSize"],
                epochs = configuration["epochSize"],
                learningRate = configuration["learningRate"],
                augmentations = configuration["augmentation"],
                momentum = 0.9,
                logsOutput = LOGS_OUTPUT
            )
    elif(configuration["model"] == "EfficientNetAdam"):
        return EfficientNetAdam(
                inputShape = configuration["imageSize"], 
                classes = 9, 
                batchSize = configuration["batchSize"],
                epochs = configuration["epochSize"],
                learningRate = configuration["learningRate"],
                augmentations = configuration["augmentation"],
                logsOutput = LOGS_OUTPUT
            )

if __name__ == '__main__':
    loadInputs()
    previousDataset = ''
    for dataset in INPUTS:
        for configuration in INPUTS[dataset]:
            if(dataset != previousDataset):
                print("LOADING DATASET:", dataset)
                X, y, y_labeled = loadDataset(dataset, configuration["imageSize"], NUMBER_PROCESSES, CLASS_SAMPLE)
            else:
                print("USING PREVIOUSLY LODADED DATASET:", dataset)

            if "hardTest" in configuration:
                print("USING HARD TEST")
                trainX, trainY = X, y_labeled
                testX, _, testY = loadDataset(configuration["hardTest"], configuration["imageSize"], NUMBER_PROCESSES)
            else:
                trainX, testX, trainY, testY = train_test_split(X, y_labeled, test_size=.3)

            print("CONFIGURATION:", configuration)
            
            start_time = time.time()
            model = getModelInstance(configuration)
            model.compileModel()

            testsResults = []
            totalTrainingTime = 0
            try:
                history = model.fit((trainX, trainY), (testX, testY))
                totalTrainingTime = (time.time() - start_time)
            except:
                print("ERROR")
            else:
                model.save(OUTPUT)
                for test in configuration["tests"]:
                    response = testModel(
                        './output/models/' + model.getName() + '.h5',
                        test["dataset"],
                        configuration["imageSize"],
                        NUMBER_PROCESSES,
                        test["classSample"]
                    )

                    testsResults.append({
                        "dataset": test["dataset"],
                        "scores": { 
                            "score": response[0][0],
                            "accuracy": response[0][1],
                        },
                        "matrix": response[1].tolist()
                    })
            finally:
                writeRecordOnReport(
                    model.getName(),
                    dataset,
                    len(X),
                    configuration["imageSize"],
                    str(datetime.timedelta(seconds = totalTrainingTime)),
                    configuration["epochSize"],
                    configuration["learningRate"],
                    configuration["batchSize"],
                    configuration["augmentation"],
                    model.getSpecialValues(configuration),
                    testsResults)
                
                del model, testsResults
                gc.collect()
                K.clear_session()
            previousDataset = dataset
