import os
import sys
import pandas as pd
import numpy as np


from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTranformationConfig:
    preprocessor_file_path=os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTranformationConfig()
        
    def get_data_transformer_object(self):
        
        '''
        This function is responsible for dat transformation
        
        '''
        
        
        try:
            numeric_columns=['reading_score', 'writing_score']
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                ]
            
            num_pipeline=Pipeline(
                [
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scalar",StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder())
                    
                ]
            )
            
            logging.info(f'Categorical Features:{categorical_columns}')
            logging.info(f'Numerical Features:{numeric_columns}')

            
            preprocessor=ColumnTransformer([
                ("numpipe",num_pipeline,numeric_columns),
                ("catpipe",cat_pipeline,categorical_columns)
            ])
            
            return preprocessor


            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    
    def initiate_data_transfromation(self,train_path,test_path):
            
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
                
            logging.info("Reading Train Test Data")

            logging.info("Obtainig preprocessor object")
                
            preprocessor_obj=self.get_data_transformer_object()
                
            target_column='math_score'
            
            train_input_feature_df=train_df.drop([target_column],axis=1)
            train_target_feature=train_df[target_column]
            
            test_input_feature_df=train_df.drop([target_column],axis=1)
            test_target_feature=train_df[target_column]
            
            logging.info("Applying Preprocessor object on training and test data!!!")

            train_input_feature_arr=preprocessor_obj.fit_transform(train_input_feature_df)
            test_input_feature_arr=preprocessor_obj.transform(test_input_feature_df)
            
            train_arr=np.c_[
                train_input_feature_arr,np.array(train_target_feature)
            ]
            
            test_arr=np.c_[
                test_input_feature_arr,np.array(test_target_feature)
            ]
            
            logging.info("Saved preprocessing object")
            
            save_obj(
                obj=preprocessor_obj,
                file_path=DataTranformationConfig.preprocessor_file_path
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )
            
            
            
                    
        except Exception as e:
            raise CustomException(e,sys)