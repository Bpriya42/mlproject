import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:
            # we get the model and preprocessor and make the prediction
            print(features)
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("load both pkl files") 
            preprocessor_obj = load_object(preprocessor_path)
            model_obj = load_object(model_path)

            print("scale the data using preprocessor")
            print(type(preprocessor_obj))
            scaled_data = preprocessor_obj.transform(features) 

            print("predict using model and preprocessed data")
            predicted = model_obj.predict(scaled_data)

            print("return predicted data")
            return predicted

        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        math_score: int, 
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.writing_score = writing_score

    def get_data_as_df(self):

        try:
            custom_data_input = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "math_score": [self.math_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input)
        
        except Exception as e:
            raise CustomException(e)

