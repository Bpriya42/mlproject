import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Split the array into train and test values")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("Set up models")
            models = {
                "Linear Regression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor()
            }

            logging.info("Evaluate model")
            model_name: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            logging.info("find the best model")
            best_model_score = max(sorted(model_name.values()))

            # mydict = {'george': 16, 'amber': 19}
            # print mydict.keys()[mydict.values().index(16)]  # Prints george
            # https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary

            best_model_name = list(model_name.keys())[ 
                list(model_name.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("no best model")
            
            logging.info("Save the model in pkl file")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info("predict the model")
            prediction = best_model.predict(X_test)

            r2_square = r2_score(y_test, prediction)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
