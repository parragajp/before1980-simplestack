import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score


def get_sorted_results(y_true, y_pred):
    return (pd.DataFrame({'Actual': y_true, 'Predicted_Prob': y_pred})
            .sort_values('Predicted_Prob', ascending=False).reset_index(drop=True))


def metrics_at_k(y_true, y_pred):
    # SLOW - find sometime to optimize this...
    results = (pd.DataFrame({'Actual': y_true, 'Predicted_Prob': y_pred})
               .sort_values('Predicted_Prob', ascending=False).reset_index(drop=True))

    # Find n first; precision
    precision_at_k = []
    recall_at_k = []
    predicted_pos_k = []
    actual_pos_k = []

    for t in results['Predicted_Prob']:
        # Need to create predicted class
        predicted_class = np.where(results['Predicted_Prob'] >= t, 1, 0)
        precision_at_k.append(precision_score(results['Actual'], predicted_class))
        recall_at_k.append(recall_score(results['Actual'], predicted_class))
        predicted_pos_k.append(((predicted_class == 1) & (results['Actual'] == 1)).sum())
        actual_pos_k.append((results['Actual'] == 1).sum())

    # Creating a dataframe of the results
    scores_at_k = pd.DataFrame({
        'n': range(1, len(results['Predicted_Prob']) + 1),
        'Recall': np.round(recall_at_k, 4),
        'Precision': np.round(precision_at_k, 4),
        'Pred_pos': predicted_pos_k,
        'Actual_pos': actual_pos_k,
        'Pred_prob': results['Predicted_Prob']
    }).melt(id_vars=['n', 'Pred_pos', 'Actual_pos', 'Pred_prob'],
            var_name='metric', value_name='score')
    return scores_at_k


def alt_avenir():
    font = "avenir"

    return {
        "config": {
            "view": {
                'continuousWidth': 'container',
                'continuousHeight': 'container'
            },
            "background": "#EEEEF8",
            "title": {
                'font': font,
                'fontSize': 20
            },
            "axis": {
                "grid": False,
                'labelFontSize': 12,
                'titleFontSize': 11,
                "labelFont": font,
                "titleFont": font
            },
            "header": {
                "labelFont": font,
                "titleFont": font
            },
            "legend": {
                "labelFont": font,
                "titleFont": font,
                "fillColor": "#EEEEF8",
                "padding": 10,
                "labelFontSize": 17,
                "titleFontSize": 17,
                "symbolStrokeWidth": 12
            }
        }
    }
