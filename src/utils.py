import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

from sklearn.metrics import r2_score

def save_obj(obj,file_path):
    
    try:
        dir_name=os.path.dirname(file_path)
    
        os.makedirs(dir_name,exist_ok=True)
    
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(xtrain,ytrain,xtest,ytest,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(xtrain,ytrain)

            model.set_params(**gs.best_params_)
            model.fit(xtrain,ytrain)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(xtrain)

            y_test_pred = model.predict(xtest)

            train_model_score = r2_score(ytrain, y_train_pred)

            test_model_score = r2_score(ytest, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)  


def load_object(file_path):
    try:
        with open(file_path,'rb') as model_obj:
            return dill.load(model_obj)
        
    except Exception as e:
        raise CustomException(e,sys)