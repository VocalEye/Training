import pandas as pd
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
import numpy as np
import gc
import multiprocessing

def imageLoader(filepath, imageSize):
    image = Image.open(filepath)
    image.load()
    
    if(imageSize[2] == 1):
        image = image.convert('L')
    image = image.resize(imageSize[:-1])

    return np.asarray(image, dtype="int32").flatten()

def divideChunks(list, chunkSize):
    for i in range(0, len(list), chunkSize):
        yield list[i:i + chunkSize]

def loadChunkFiles(index, dataset, chunk, imageSize, sendEnd):
    data = []
    for filename in chunk:
        file = os.path.join(dataset, filename)
        number, quadrant = map(int, filename[:-4].split('_'))

        data.append([
            number, 
            imageLoader(file, imageSize), 
            quadrant - 1
        ])
    sendEnd.send(data)

def datasetLoader(dataset, imageSize, numberProcesses):
    data = []
    filenames = os.listdir(dataset)
    size = len(filenames)
    chunkSize = int(size/numberProcesses)

    chunks = list(divideChunks(filenames, chunkSize))
    if len(chunks) > numberProcesses:
        chunks.pop()
    
    print("LOADING DATASET WITH PROCESSES #", len(chunks))
    jobs = []
    pipeList = []
    for index, chunk in enumerate(chunks):
        receiverEnd, senderEnd = multiprocessing.Pipe(False)
        process = multiprocessing.Process(target = loadChunkFiles, args = (index, dataset, chunk, imageSize, senderEnd))
        jobs.append(process)
        pipeList.append(receiverEnd)
        process.start()
    
    resultsList = [x.recv() for x in pipeList]
    
    for process in jobs:
        process.join()
    
    for result in resultsList:
        data.extend(result)

    print("DATASET LOADED - IMAGES LOADED:", len(data))
    return data

def loadDataset(dataset, imageSize, numberProcesses, classSample = None):
    df = pd.DataFrame(
        datasetLoader(dataset, imageSize, numberProcesses), 
        columns=['number', 'image', 'quadrant']
    )

    df = df.sample(frac=1).reset_index(drop=True)
    if classSample is not None:
        df = df.groupby('quadrant').apply(lambda x: x.sample(classSample))

    X = df.image.to_numpy()
    y = df.quadrant.to_numpy()

    del df
    gc.collect()

    X = np.asarray([ np.asarray(x.reshape(imageSize)) for x in X ])
    y = np.asarray([ np.asarray(cls.reshape(1)) for cls in y ])

    X = X/255
    y_labeled = to_categorical(y, 9)

    print("TOTAL LOAD:", len(X))

    return X, y, y_labeled