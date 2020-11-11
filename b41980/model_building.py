# Loading in the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score)  # Performance Metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# import altair as alt
from joblib import dump

# Loading in data
ml_data = pd.read_csv(
    "https://raw.githubusercontent.com/byuidatascience/data4dwellings/master/data-raw/dwellings_ml/dwellings_ml.csv")

# Doing a quick look
# ml_data.info()

# Features and target data
X = ml_data.loc[:, ["livearea", "stories", "numbdrm", "numbaths"]]
y = ml_data.before1980  # variable we are trying to predict

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=74)

# Checking out what train test split did
# print(f"Rows in X: {X.shape[0]}")
# print(f"X_train Rows in X_train: {X_train.shape[0]}")
# print(f"X_test Rows in X_test: {X_test.shape[0]}")

# # Creating the Decision Tree Model
# tree_clf = DecisionTreeClassifier()
# tree_clf.fit(X_train, y_train)  # Fitting model with the training data
# y_pred = tree_clf.predict(X_test)  # Predicting on the test data

# # Evaluating Performance of Model
# accuracy_score(y_test, y_pred)
# recall_score(y_test, y_pred)
# precision_score(y_test, y_pred)

# Creating the Random Forest Model
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)  # Fitting model with the training data
y_pred2 = rf_clf.predict(X_test)  # Predicting on the test data

# Evaluating Performance of Model
accuracy_score(y_test, y_pred2)  # 82% acccuracy
recall_score(y_test, y_pred2)
precision_score(y_test, y_pred2)

# Printing Tree
# tree_clf = DecisionTreeClassifier(max_depth=2)
# tree_clf.fit(X_train, y_train)
# plot_tree(tree_clf, feature_names=X_train.columns, filled=True, class_names=["Built Before 1980", "Built After 1980"])

# # Exploring model feature importances
# tree_clf_feat_import = (pd.DataFrame({
#     "Feature Importances": tree_clf.feature_importances_,
#     "Feature Names": X_train.columns,
# }).sort_values("Feature Importances", ascending=False).head(20))

# base_fi = (alt.Chart(tree_clf_feat_import).encode(
#     x=alt.X("Feature Importances"),
#     y=alt.Y("Feature Names", sort="-x")
# ).properties(title={"text": "tree Classifier Feature Importances"}))

# chart = (base_fi.mark_bar(color="navy", size=2) +
#          base_fi.mark_circle(color="red", size=100, opacity=1)).configure_axis(
#     labelFontSize=12).configure_title(fontSize=25)

# Saving model as a pickle file
dump(rf_clf, 'rf_before1980.joblib')
