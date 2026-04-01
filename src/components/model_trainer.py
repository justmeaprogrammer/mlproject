import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
) 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test inout data")
            xtrain,xtest,ytrain,ytest=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            
            models={
                "Linear Regression":LinearRegression(),
                "K-Nearest Neighbours":KNeighborsRegressor(),
                "Decison Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "XG Boost":XGBRegressor(),
                "Catplot Boosting":CatBoostRegressor()
            }
            
            model_report:dict =evaluate_model(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,
                                              models=models)
            
            # Getting best model  score
            best_modal_score=max(sorted(model_report.values()))
            
            #Getting its index and then the name of the model
            best_modal_name=list(model_report.keys())[
                list(model_report.values()).index(best_modal_score)
                ]
            best_model=models[best_modal_name]
            
            if best_modal_score<0.6:
                raise CustomException("NO BEST MODEL FOUND!!!")
            
            logging.info("Best model found")
            
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(xtest)
            score=r2_score(ytest,predicted)
            
            return score
            
            
        except Exception as e:
            raise CustomException(e,sys) 