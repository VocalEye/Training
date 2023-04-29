
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from src.utils.loadDataset import loadDataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

IMAGE_SIZE = (32, 32, 3)

NUMBER_PROCESSES = 10
DATASET = './input/test/Real Features 2/'
MODEL = './output/models/AlexNet-20230420-234554.h5'
CLASS_SAMPLE= None
def getPredictedClass(predictions):
    y_pred_class = []

    for prediction in predictions:
        search = np.where(prediction == np.amax(prediction))[0]
        if not search.size:
            return None
        y_pred_class.append(search[0])
    
    return y_pred_class

def testModel(modelName, dataset, imageSize, numberProcesses, classSample = None):
    X_test, y_test, y_labeled = loadDataset(dataset, imageSize, numberProcesses, classSample)
    model = load_model(modelName)
    score = model.evaluate(X_test, y_labeled)
    if np.isnan(score[0]):
        score[0] = 0
    y_pred = getPredictedClass(model.predict(X_test))
    
    confusion_mtx = np.array([]) if y_pred is None else confusion_matrix(y_test, y_pred)
    return [score, confusion_mtx]

if __name__ == '__main__':
    response = testModel(MODEL, DATASET, IMAGE_SIZE, NUMBER_PROCESSES, CLASS_SAMPLE)

    print('Test Score = ', response[0][0])
    print('Test Accuracy = ', response[0][1])
    print(response[1])
