"""the purpose of this file is to load the data from load_data.py 
and load and trained model from tarin_model.py and save it to a file"""

import joblib
from data.load_data import load_data
from model.train_model import train_model

def train_and_save_model():
    """Load the data , train the model, and save it to file"""
    df = load_data() # this function provides the data frame

    # train the model
    model = train_model(df) # this function provides the trained model

    #save the train model to file 
    joblib.dump(model, "random_forest_model.pkl")
    
    print("model saves to random_forest_model.pkl")

if __name__ =='__main__':
    train_and_save_model()    