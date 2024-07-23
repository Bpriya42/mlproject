import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    # we are gonna make a pkl file to sstore a preprocessor

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        The feature data will get transformed using the necessary pipelines
        '''
        try:
            # Separate the features into numerical and categorical columns
            num_col = ["math score", "writing score"]

            cat_col = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]

            # make two pipelines, one for numerical and one for categorical
            num_pipeline = Pipeline (
                steps= [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler)
                ]
            )

            cat_pipeline = Pipeline (
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oh_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean = False))
                ]
            )

            # make the preprocessor
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, num_col)
                    ("categorical_pipeline", cat_pipeline, cat_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function starts the data transformation
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()

            target_col = "reading score"
            numerical_col = ["math score", "writing score"]

            input_feature_train = train_df.drop(columns=[target_col], axis = 1)
            target_feature_train = train_df[target_col]

            input_feature_test = test_df.drop(columns=[target_col], axis = 1)
            target_feature_test = test_df[target_col]

        except Exception as e:
            raise CustomException(e, sys)