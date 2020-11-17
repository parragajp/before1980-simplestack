# Loading in the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from joblib import dump

# Loading in data
ml_data = pd.read_csv(
    "https://raw.githubusercontent.com/byuidatascience/data4dwellings/master/data-raw/dwellings_ml/dwellings_ml.csv")

# Doing a quick look
# ml_data.info()

# # Features and target data
X = ml_data.loc[:, ["livearea", "numbdrm", "numbaths", "arcstyle_ONE-STORY", "gartype_Att", "basement"]]
y = ml_data.before1980  # variable we are trying to predict

# # Features and target data - Old model
# X = ml_data.loc[:, ["livearea", "stories", "numbdrm", "numbaths"]]
# y = ml_data.before1980  # variable we are trying to predict

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=74)

# Creating the Random Forest Model
rf_clf = RandomForestClassifier(max_depth=30)
rf_clf.fit(X_train, y_train)  # Fitting model with the training data
y_pred = rf_clf.predict(X_test)  # Predicting on the test data
y_pred_prob = rf_clf.predict_proba(X_test)  # Predicting on the test data

# Saving model as a pickle file
dump(rf_clf, 'b41980/model/rf_before1980.joblib')

# Saving data to zipped npz file
np.savez_compressed("b41980/data/modelling_data.npz",
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    y_pred=y_pred,
                    y_pred_proba=y_pred_prob)
