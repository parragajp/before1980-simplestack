# Loading in the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import altair as alt
from joblib import load, dump
from b41980.ml_helpers import metrics_at_k, alt_avenir

# Altair settings
alt.themes.register('alt_avenir', alt_avenir)
alt.themes.enable('alt_avenir')
alt.data_transformers.disable_max_rows()

#### Seed averaging to get a range of metrics.. ####

# Loading in data
ml_data = pd.read_csv(
    "https://raw.githubusercontent.com/byuidatascience/data4dwellings/master/data-raw/dwellings_ml/dwellings_ml.csv")

# # Features and target data
X = ml_data.loc[:, ["livearea", "numbdrm", "numbaths", "arcstyle_ONE-STORY", "gartype_Att", "basement"]]
y = ml_data.before1980  # variable we are trying to predict

# Seed averaging to get some faithful evaluation metrics
accuracy, precision, recall = [], [], []
for seed in np.arange(24, 50):

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed)

    # Creating the Random Forest Model
    rf_clf = RandomForestClassifier(max_depth=30)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    y_pred_prob = rf_clf.predict_proba(X_test)

    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))

pd.Series(accuracy).agg([min, max])
pd.Series(precision).agg([min, max])
pd.Series(recall).agg([min, max])

#### Making model evaluation charts ####

# Loading in simple model and data previously built for simplicity
# This model will be used from here on out.
# If trying to improve consistency of model might want to take an ensemble of
# the same model trained on different seeds of data
rf_clf = load("b41980/model/rf_before1980.joblib")
data = np.load("b41980/data/modelling_data.npz", allow_pickle=True)
X_train, y_train, X_test, y_test, y_pred, y_pred_proba, column_names = [data[name] for name in data.files]
column_names_aes = ["Liveable Sqft", "Bedrooms", "Baths", "One Story Style",
                    "Attached Garage", "Basement"]


# Exploring model feature importances
rf_clf_feat_import = (pd.DataFrame({
    "Feature Importances": np.round(rf_clf.feature_importances_, 2),
    "Feature Names": column_names_aes,
}).sort_values("Feature Importances", ascending=False).head(20))

base_fi = (alt.Chart(rf_clf_feat_import).encode(
    x=alt.X("Feature Importances:Q", title=""),
    y=alt.Y("Feature Names:N", title="", sort="-x"),
    tooltip=[alt.Tooltip("Feature Importances:Q", title='Feature Importance')]
).properties(title={"text": "Feature Importances"}))

chart = (base_fi.mark_bar(color="#A9A9A9", size=4) +
         base_fi.mark_circle(color="#FF6700", size=120, opacity=1)).configure_axis(
    labelFontSize=15,
    titleFontSize=14,
).interactive()
chart.save("b41980/static/feat_import_spec.json")

# Precision Recall Chart
scores_at_k = metrics_at_k(y_test, y_pred_proba[:, 1])
alt_thresh = alt.Chart(scores_at_k).mark_line(strokeWidth=4).encode(
    alt.X("Pred_prob:Q", title='Threshold'),
    alt.Y("score:Q", title='Scores'),
    alt.Color('metric:N', title='', scale=alt.Scale(range=['#FF6700', '#A9A9A9'])),
    tooltip=[
        alt.Tooltip('metric', title='Metric'),
        alt.Tooltip('Pred_pos:O', title='# of correctly predicted houses built before 1980'),
        alt.Tooltip('n', title='# of predicted houses built before 1980'),
        alt.Tooltip('Actual_pos:O', title='Total houses built before 1980'),
        alt.Tooltip('score', title='Score'),
        alt.Tooltip('Pred_prob', title='Threshold')
    ]
).properties(title={"text": "Precision Recall Curves"},
             width=360, height=300
             ).interactive().configure_legend(
    symbolStrokeWidth=12,
    fillColor="#EEEEF8",
    padding=10,
    labelFontSize=17
)
alt_thresh.save("b41980/static/pr_spec.json")

# # TSNE Chart
# pipe = make_pipeline(
#     StandardScaler(),
#     TSNE(n_components=2)
# )
# # Fitting pipeline
# components = pipe.fit_transform(X_train)
# # pipe['tsne'].explained_variance_
# pipe.transform(X_test)
# # Creating dataframe
# tsne_data = pd.DataFrame(components, columns=['pca1', 'pca2'])
# tsne_data['target'] = np.where(y_train == 1, "Yes", "No")
# tsne_data.to_json("tsne_data.json")

# tsne_chart = alt.Chart(tsne_data).mark_circle(opacity=.4).encode(
#     alt.X("pca1", title="principal component 1"),
#     alt.Y("pca2", title="principal component 2"),
#     alt.Color("target:N", title='Built before 1980', scale=alt.Scale(range=["gray", "orange"]))
# ).properties(**props).configure_axis(
#     labelFontSize=12, titleFontSize=11
#     )

# # Saving model to file
# dump(pipe, 'b41980/model/tsne.joblib')
# # Saving chart to file -- probably will be building this inside the app though...
# tsne_chart.save("b41980/static/tsne_spec.json")

# Isomap Chart
# pipe = make_pipeline(
#     StandardScaler(),
#     Isomap(n_components=2)
# )
# # Fitting pipeline
# components = pipe.fit_transform(X_train)
# # pipe['Isomap'].explained_variance_
# pipe.transform(X_test)
# # Creating dataframe
# isomap_data = pd.DataFrame(components, columns=['pca1', 'pca2'])
# isomap_data['target'] = np.where(y_train == 1, "Yes", "No")
# isomap_data.to_json("isomap_data.json")

# isomap_chart = alt.Chart(isomap_data).mark_circle(opacity=.4).encode(
#     alt.X("pca1", title="principal component 1"),
#     alt.Y("pca2", title="principal component 2"),
#     alt.Color("target:N", title='Built before 1980', scale=alt.Scale(range=["gray", "orange"]))
# ).properties(**props).configure_axis(
#     labelFontSize=12, titleFontSize=11
# ).interactive()
# isomap_chart

# # Saving model to file
# dump(pipe, 'b41980/model/isomap.joblib')

# Locally Linear Embedding Chart
pipe = make_pipeline(
    StandardScaler(),
    LocallyLinearEmbedding(n_components=2, eigen_solver='dense')
)
# Fitting pipeline
components = pipe.fit_transform(X_train)
# pipe['LocalLinearEmbedding'].explained_variance_

# Creating dataframe
lle_data = pd.DataFrame(components, columns=['pca1', 'pca2'])
lle_data['target'] = np.where(y_train == 1, "Yes", "No")
lle_data.to_json("b41980/data/lle_chart_data.json")

lle_chart = alt.Chart(lle_data).mark_circle(opacity=.4).encode(
    alt.X("pca1", title="principal component 1"),
    alt.Y("pca2", title="principal component 2"),
    alt.Color("target:N", title='Built before 1980', scale=alt.Scale(range=["gray", "orange"]))
).properties().configure_axis(
    labelFontSize=12, titleFontSize=11
).interactive()
lle_chart

# Saving model to file
dump(pipe, 'b41980/model/lle.joblib')


#### Shapley Values to interpret model further ####

# Checking out the SHAP values to explain model features.
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_train)
shap_plot = shap.summary_plot(shap_values, X_train)
shap_values
