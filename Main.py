#suicide prediction program
# import time
'''
suicide prediction program is working 
            but 
will take time so dont quit in middle
'''

# import all parts as module from src
from src import DataProcessing
from src.AccuracyBarGraph import AccuracyPlot
from src.CorrelationMatrix import CorrMatrix
from src.DataSplitting import DataSplit
from src.FeatureImportance import featuring_importance
# import src.TuningWithGridSearchCV as gscv
# import src.TuningWithRandomizedSearchCV as rscv
import src.GridSearchCVpickle as prod_gscv
import src.RandomizedSearchCVpickle as prod_rscv
import src.TuningWithGridSearchCV as train_gscv
import src.TuningWithRandomizedSearchCV as train_rscv
import src.output as output

#from src.DnnClassifier import tensorflow_dnn as dnn
# from src.AccuracyBarGraph import AccuracyPlot


# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json


def suicide():
    # start = time.time()

    # data loading, checking, cleaning and encoding
    data = DataProcessing.process()

    '''
    - Data Cleaning nd Encoding
    - Corrlation Matrix
    - Splitting the data into training and testing
    - Feature importance
    '''
    #creating the correlation matrix
    CorrMatrix(data)


    #splitting the dataset into train and test sets
    X, y, X_train, X_test, y_train, y_test = DataSplit(data)


    #visualising the feature importance
    featuring_importance(X, y)

    #Dictionary to store accuracy results of different algorithms
    accuracyDict = {}

    #Dictionary to store time log of different funcitons
    timelog = {}

    '''
    - Tuning
    '''

    # Only prediction
    prod_gscv.GridSearch(X_train, X_test, y_train, y_test, accuracyDict, timelog)

    # Only prediciton
    prod_rscv.RandomizedSearch(X_train, X_test, y_train, y_test, accuracyDict, timelog)

    # Tuning with GridSearchCV
    train_gscv.GridSearch(X_train, X_test, y_train, y_test, accuracyDict, timelog)

    # Tuning with RandomizedSearchCV
    train_rscv.RandomizedSearch(X_train, X_test, y_train, y_test, accuracyDict, timelog)

    
    #DNN implimentation 
    #dnn(data, X_train, X_test, y_train, y_test, accuracyDict, timelog)

    # #Generating PDF reports
    output.get_pdf_output()

    print("accuracyDict:\n")
    print(json.dumps(accuracyDict, indent=1))
    
    '''
    - Accuracy Bar Graph
    '''
    AccuracyPlot(accuracyDict)
    '''
    - Modelling
    '''

    # end = time.time()
    print("Time Taken by Grid Search: ", timelog['GridSearch models'],"Seconds")
    #print("Time Taken by Random Search: ", timelog['Randomized models'],"Seconds")
    #print("Time Taken by DNN Classifier: ", timelog['DNN Classifier'],"Seconds")
    # print("Total Time taken: ", end - start,"seconds")

# suicide ()