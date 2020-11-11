# Importing modules
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import load

# Loading in our trained model
rf = load("model/rf_before1980.joblib")

# Creating random feature data
abstr = 200
livearea = 200
finishedbase = 200
base = 200
units = 200
stories = 200
cars = 200
bdrms = 200
baths = 200

# Creating feature array
new_instance = np.array([[abstr, livearea, finishedbase,
                          base, units, stories, cars,
                          bdrms, baths]])

# Predicting new_instance target
prediction = rf.predict(new_instance)

# Returning results back
f"Predicted value for {new_instance} is: {prediction}"
