import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def read_data():
    static_info = pd.read_csv('info_static.csv', index_col=0, sep=',')

    y_train = pd.read_csv('ytrain_NpxebDC.csv', sep=',', index_col=0)
    y_test_isna = pd.read_csv('ytest_isna.csv', sep=',', index_col=0)
    x_train = pd.read_csv('xtrain.csv', sep=',', index_col=0)
    x_test = pd.read_csv('xtest.csv', sep=',', index_col=0)
    return y_train, x_test, y_test_isna, x_train, static_info


def create_df_train(data_):
    # Drop NaN and change text to number
    df_ = data_.copy()

    df_ = df_.dropna()

    for i in df_.columns:
        df_[i] = df_[i].map({'Offline':0, 'Down':1, 'Available':2, 'Passive':3, 'Charging':4})
    return df_


def create_y(df):
    # Create Dataframe with columns ['month', 'day', 'dayofweek', 'hour', 'minute']
    # from index of df
    data_ = []
    for times in df.index:
        time = pd.Timestamp(times)
        month = time.month
        day = time.day
        dayofweek = time.dayofweek
        hour = time.hour
        minute = time.minute
        data_.append([month, day, dayofweek, hour, minute])
    y_ = pd.DataFrame(np.array(data_), columns=['month', 'day', 'dayofweek', 'hour', 'minute'], index=df.index)
    return y_



from sklearn.ensemble import RandomForestClassifier

def train_and_predict(train_, target_, test_):
    # train and predict for 1 column
    clf_= RandomForestClassifier(random_state=42, n_estimators= 100)
    clf_.fit(train_, target_)
    pred_ = clf_.predict(test_)
    return pred_


def write_pred(pred_, test_):
    df = test_.copy()
    for i in range(pred_.shape[1]):
        df[df.columns[i]] = pred_[:, i]
    for i in df.columns:
        df[i] = df[i].map({0:'Offline', 1:'Down', 2:'Available', 3:'Passive', 4:'Charging'})
    return df



def plot_pred(train_, test_, df_pred_, col_):
    plt.plot(train_.index, train_[col_])
    plt.plot(test_.index, df_pred_[col_])
    plt.show()


# Get the data
data_ytrain, data_xtest, data_ytest_isna, data_xtrain, data_static_info = read_data()
test_index = np.array(data_ytest_isna.index)

# Clean data and map str --> int
# # TO DO : use static info and average_flow in xtest
df_train = create_df_train(data_ytrain)
df_test = pd.DataFrame([], index=test_index, columns=df_train.columns)

# Change format of input (Timestamp --> int[])
train = create_y(df_train)
test = create_y(df_test)

# Predict using random forest
target = df_train
predictions = train_and_predict(train, target, test)

# Map back int --> str  and write results in a Dataframe
df_pred = write_pred(predictions, df_test)
df_pred.to_csv("y_random.csv")