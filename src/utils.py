import os
import sys
import pandas as pd
import numpy as np
import dill
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
    
def evaluate_model(xtrain,ytrain,xtest,ytest,models):
    try:
        report={}
        for i in range(len(list(models))):
            
            model=list(models.values())[i]
        
            model.fit(xtrain,ytrain)
        
            ytestpred=model.predict(xtest)
        
            score=r2_score(y_true=ytest,y_pred=ytestpred)
        
            report[list(models.keys())[i]]=score
        
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
        
        
        