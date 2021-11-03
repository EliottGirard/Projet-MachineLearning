"""A module defining the custom metric.

Custom metric script for the DataChallenge ENS 2021
Challenge from Planete OUI
Author: Aude Laurent, aude.laurent@planete-oui.fr

The custom metric script must contain the definition of custom_metric_function
and a main function
that reads the two csv files with pandas and evaluate the custom metric.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def custom_metric_function(dataframe_y_true, dataframe_y_pred):
    """Compute the custom metric.

    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y. This dataframe was
            obtained by reading a csv file with following instruction:
            dataframe_y_true = \
                pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following
            instruction:
            dataframe_y_pred = \
                pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.

    Method:

    Shape of dataframe_y:
        - T rows. 1 row per timestamps. From start date/time to end date/time
            with 15 min step
        - n_terminals columns. 1 column per terminal.

    Steps:
        1. For each terminal & for each class, the F1 score is computed
        2. For each class, the F1 over all terminals is computed
            2a. Sum over all terminals
            2b. Divide by number of terminals
        3. The weighted average F1 over the classes is computed
            3a. Weight the F1 class with the frequency of the class
            3b. Divide by the number of values in the dataframe

    """
    labels_frequency = {"Available": 0.593729, "Charging": 0.06012988,
                        "Passive": 0.06959437, "Offline": 0.17697645,
                        "Down": 0.08682056}
    labels = list(labels_frequency.keys())
    freqs = np.array(list(labels_frequency.values()))
    freqs = freqs / np.sum(freqs)

    score = [
        _compute_score_terminal(dataframe_y_true[terminal],
                                dataframe_y_pred[terminal],
                                labels, freqs)
        for terminal in dataframe_y_true.columns
    ]
    score = np.mean(score)

    return score


def _compute_score_terminal(serie_y_true, serie_y_pred, labels, freqs):
    # Retrieve non NaN values
    notnull = serie_y_true.notnull()
    idx = serie_y_true.index[notnull].tolist()
    if not(idx):  # whole column is NaN: no prediction is expected
        return 1.

    terminal_y_pred_nonnan = serie_y_pred.loc[idx]
    terminal_y_true_nonnan = serie_y_true.loc[idx]

    # Step 1.
    terminal_f1s = f1_score(terminal_y_true_nonnan, terminal_y_pred_nonnan,
                            labels=labels, average=None)
    # replace argument 'zero_division=1' for version compatibility
    obs_labels_true = np.unique(terminal_y_true_nonnan.values)
    obs_labels_pred = np.unique(terminal_y_pred_nonnan.values)
    for i, label in enumerate(labels):
        if label not in obs_labels_true and label not in obs_labels_pred:
            terminal_f1s[i] = 1.0

    # Step 2a.
    col_score = np.sum(terminal_f1s * freqs)
    return col_score


if __name__ == '__main__':
    # CSV_FILE_Y_TRUE = '--------.csv'
    # CSV_FILE_Y_PRED = '--------.csv'
    CSV_FILE_Y_TRUE = 'ytest.csv'
    CSV_FILE_Y_PRED = 'yrandom.csv'

    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    print(custom_metric_function(df_y_true, df_y_true))
    print(custom_metric_function(df_y_true, df_y_pred))
