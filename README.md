An end-to-end Student Performance Indicator. 

The model determines the Reading Score of the students based on their other scores (Math and English) as well as other information like parental level of income, gender etc. The project consists of the following files:
1. data ingestion - Initialises the data and creates train and test datasets
2. data transformation -  Transforms the data by scaling numerical values and one hot encoding the categorical values. 
3. model training - This file chooses the best regression model ( Linear Regression, KNeighborsRegressor, DecisionTreeRegressor ) and returns the best model for use (as long as it has an R-score above 0.6 )
4. Training pipeline
5. Prediction pipeline
6. A Flask application to present the model and present predicitions based on the user input

How to run the file

1. Clone this repository
2. Complete downloading the requirements by running: pip install -r requirements.txt
3. Run the app.py file: python app.py
