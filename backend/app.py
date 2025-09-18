from flask import Flask , request, jsonify
# Flask is to creat an api end point 
# request is to get the data from the streamlit app
# jsonify is to convert the data into jason formate and return it to streamlit app
import joblib 
import pandas as pd

app = Flask (__name__) # creating an object of the flask

# Load the trained model
model = joblib.load("random_forest_model.pkl")

@app.route("/predict", methods=["POST"]) #
def predict():
    """API endpoint to get prediction from the trained model."""
    data = request.json # Get data from the request {"total_bill": 16.99, "sex":"Male"...}

    input_df = pd.DataFrame([{"total_bill":data["total_bill"],
                              "sex":data["sex"],
                              "smoker":data["smoker"],
                              "day":data["day"],
                              "time":data["time"],
                              "size":data["size"]}])
    #covert data to data frame
    prediction = model.predict(input_df) # Make prediction # array [tip_value]
    return jsonify({"prediction": prediction[0]}) #return prediction as json

if __name__ == "__main__":
    app.run(debug=True) # run the flask app