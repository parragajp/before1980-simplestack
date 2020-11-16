# Loading in the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import altair as alt
from joblib import dump
from b41980.ml_helpers import metrics_at_k
alt.data_transformers.disable_max_rows()
props = {'width': 550, 'height': 400}

# Loading in data
ml_data = pd.read_csv(
    "https://raw.githubusercontent.com/byuidatascience/data4dwellings/master/data-raw/dwellings_ml/dwellings_ml.csv")

# Doing a quick look
# ml_data.info()

# # Features and target data
X = ml_data.loc[:, ["livearea", "numbdrm", "numbaths", "arcstyle_ONE-STORY", "gartype_Att", "basement"]]
y = ml_data.before1980  # variable we are trying to predict

# # Features and target data
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

# Evaluating Performance of Model
accuracy_score(y_test, y_pred)  # % acccuracy
scores_at_k = metrics_at_k(y_test, y_pred_prob[:, 1])

# Exploring model feature importances
rf_clf_feat_import = (pd.DataFrame({
    "Feature Importances": rf_clf.feature_importances_,
    "Feature Names": X_train.columns,
}).sort_values("Feature Importances", ascending=False).head(20))

base_fi = (alt.Chart(rf_clf_feat_import).encode(
    x=alt.X("Feature Importances:Q"),
    y=alt.Y("Feature Names:N", sort="-x")
).properties(title={"text": "rf Classifier Feature Importances"}))

chart = (base_fi.mark_bar(color="navy", size=2) +
         base_fi.mark_circle(color="'#FF6700'", size=100, opacity=1)).configure_axis(
    labelFontSize=12).configure_title(fontSize=25)

chart2 = base_fi.mark_bar(color="navy", size=2)

# Printing chart
# chart2

# Precision Recall Chart
alt_thresh = alt.Chart(scores_at_k).mark_line(strokeWidth=4).encode(
    alt.X("Pred_prob:Q", title='Threshold'),
    alt.Y("score:Q", title='Scores'),
    alt.Color('metric:N', title='', scale=alt.Scale(range=['#FF6700', '#A9A9A9'])),
    tooltip=[
        alt.Tooltip('metric', title='Metric'),
        alt.Tooltip('Pred_pos:O', title='# of correctly predicted homes built before 1980'),
        alt.Tooltip('n', title='# of predicted houses built before 1980'),
        alt.Tooltip('Actual_pos:O', title='Total homes built before 1980'),
        alt.Tooltip('score', title='Score'),
        alt.Tooltip('Pred_prob', title='Threshold')
    ]
).properties(**props, title={"text": "Precision Recall Curves"}).interactive().configure_legend(
    symbolStrokeWidth=10,
    fillColor="white",
    padding=10,
    labelFontSize=15
).configure(background="#EEEEF8").configure_axis(grid=False)

alt_thresh.save("b41980/templates/precision_recall_chart.html")

# Saving model as a pickle file
dump(rf_clf, 'rf_before1980.joblib')
