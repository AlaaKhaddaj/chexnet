import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

#--------------------------------------------------------------------------------

def main ():

    # runTest()
    runTrain()

#--------------------------------------------------------------------------------

def runTrain():

    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    RESNET18 = "RES-NET-18"
    RESNET50 = "RES-NET-50"

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    #---- Path to the directory with images
    pathDirData = './database'

    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'

    #---- Neural network parameters: type of the network, is it pre-trained
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    # nnArchitecture = RESNET18
    # nnArchitecture = RESNET50
    nnIsTrained = True
    nnClassCount = 14

    #---- Training settings: batch size, maximum number of epochs
    # trBatchSize = 16
    trBatchSize = 16
    trMaxEpoch = 100

    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 128
    imgtransCrop = 224

    pathModel = 'm-' + timestampLaunch + '.pth.tar'

    print ('Training NN architecture = ', nnArchitecture)

    print("Training Parameters:")
    print("Batch Size:", trBatchSize)
    print("Number of Epochs:", trMaxEpoch)

    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)

    trBatchSize = 16

    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#--------------------------------------------------------------------------------

def runTest():

    pathDirData = './database'
    pathFileTest = './dataset/test_1.txt'
    nnArchitecture = 'DENSE-NET-121'
    # nnArchitecture = 'RES-NET-18'
    # nnArchitecture = 'RES-NET-50'
    # nnArchitecture = 'DENSE-NET-169'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'm-25012018-123527.pth.tar'
    # pathModel = 'm-21032021-160042.pth.tar'

    timestampLaunch = ''

    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#--------------------------------------------------------------------------------

if __name__ == '__main__':
    main()





