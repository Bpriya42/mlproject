import os
import sys

import numpy as np
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    '''Function to create a pkl file with object'''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException( e, sys)
    
def load_object(filepath):
    '''Function to load and return created pkl file'''
    try:
        with open(filepath, "rb") as file_obj:
            print("loading pkl file")
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    '''Function to evaluate the model based on input data'''
    try:
        model_list = {}

        for i in range(len(list(models))):

            logging.info("Start evaluations")
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            logging.info("# make predictions")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            logging.info("Evaluate") 

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            model_list[list(models.keys())[i]] = test_r2

        return model_list

    except Exception as e:
        raise CustomException(e,sys)