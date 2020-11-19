# Loading in the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import altair as alt
from joblib import load, dump
from b41980.ml_helpers import metrics_at_k, alt_avenir
import shap
shap.initjs()

# Altair settings
alt.themes.register('alt_avenir', alt_avenir)
alt.themes.enable('alt_avenir')
alt.data_transformers.disable_max_rows()
props = {'width': 400, 'height': 300}


# Loading in data and model
rf_clf = load("b41980/model/rf_before1980.joblib")
data = np.load("b41980/data/modelling_data.npz", allow_pickle=True)
X_train, y_train, X_test, y_test, y_pred, y_pred_proba, column_names = [data[name] for name in data.files]
column_names_aes = ["Liveable Sqft", "Bedrooms", "Baths", "One Story Style",
                    "Attached Garage", "Basement"]

"""
Evaluating performance scores. 
I will probably put this on the about page of the website. As a panel on
the right side maybe?
"""
accuracy_score(y_test, y_pred)  # .902967 acccuracy
precision_score(y_test, y_pred)  # .925699 precision
recall_score(y_test, y_pred)  # .918144

"""
Charts exploring feature importances and precision recall curve.
Knn like chart with principal components being each axis
"""
# Exploring model feature importances
rf_clf_feat_import = (pd.DataFrame({
    "Feature Importances": np.round(rf_clf.feature_importances_, 2),
    "Feature Names": column_names_aes,
}).sort_values("Feature Importances", ascending=False).head(20))

base_fi = (alt.Chart(rf_clf_feat_import).encode(
    x=alt.X("Feature Importances:Q", title=""),
    y=alt.Y("Feature Names:N", title="", sort="-x"),
    tooltip=[alt.Tooltip("Feature Importances:Q", title='Feature Importance')]
).properties(title={"text": "Feature Importances"},
             width=300, height=400))

chart = (base_fi.mark_bar(color="#A9A9A9", size=4) +
         base_fi.mark_circle(color="#FF6700", size=120, opacity=1)).configure_axis(
    labelFontSize= 15,
    titleFontSize= 14,
         ).interactive()

chart.save("b41980/static/feat_import.json")

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
).properties(**props, title={"text": "Precision Recall Curves"}).interactive().configure_legend(
    symbolStrokeWidth=12,
    fillColor="#EEEEF8",
    padding=10,
    labelFontSize=17
)
alt_thresh
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
pipe = make_pipeline(
    StandardScaler(),
    Isomap(n_components=2)
)
# Fitting pipeline
components = pipe.fit_transform(X_train)
# pipe['Isomap'].explained_variance_
pipe.transform(X_test)
# Creating dataframe
isomap_data = pd.DataFrame(components, columns=['pca1', 'pca2'])
isomap_data['target'] = np.where(y_train == 1, "Yes", "No")
isomap_data.to_json("isomap_data.json")

isomap_chart = alt.Chart(isomap_data).mark_circle(opacity=.4).encode(
    alt.X("pca1", title="principal component 1"),
    alt.Y("pca2", title="principal component 2"),
    alt.Color("target:N", title='Built before 1980', scale=alt.Scale(range=["gray", "orange"]))
).properties(**props).configure_axis(
    labelFontSize=12, titleFontSize=11
).interactive()
isomap_chart

# Saving model to file
dump(pipe, 'b41980/model/isomap.joblib')


"""
Shapley Values to interpret model further
"""

# Checking out the SHAP values to explain model features.
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_train)
shap_plot = shap.summary_plot(shap_values, X_train)
shap_values


alt.Chart("b41980/static/tsne.json").mark_circle()
