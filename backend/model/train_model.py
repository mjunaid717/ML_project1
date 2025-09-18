import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib #for saving the model

def train_model(df):
    """Train a machine learing model on the provided DataFrame"""

    #separate tips and rest of the variables as x and y
    x = df.drop("tip",axis=1)
    y = df["tip"]

    #Identify catagorical and numerical columns
    categorical_cols = x.select_dtypes(include=["category"]).columns
    numerical_cols = x.select_dtypes(exclude=["category"]).columns

    #one hot encoding for categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('num','passthrough', numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols),
            ]
    )
    #creating a pipeline that first transform the data then fits the model
    model = Pipeline(
        steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100,random_state=42)),
        ]
    )
    # Split the data into training and testing sets
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit the model 
    model.fit(x_train, y_train)

    return model

    