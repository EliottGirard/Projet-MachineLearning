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


def create_df_ytrain(data_):
    # Change str to int and fill NaN
    df_ = data_.copy()

    for col in df_.columns:
        if df_[col].to_list().count('Down')/len(df_[col]) > 0.8:
            df_[col] = df_[col].fillna('Down')
        else:
            df_[col] = df_[col].fillna('Available')

    for col in df_.columns:
        df_[col] = df_[col].map({'Offline':0, 'Down':1, 'Available':2, 'Passive':3, 'Charging':4})
    return df_


def create_df_x(data_):
    df_ = data_.copy()
    df_average = np.average(df_['average_flow'].dropna())
    df_ = df_['average_flow'].fillna(df_average)
    return pd.DataFrame(df_)


def define_test(data_ytest_, data_ytrain_, data_xtest_, data_xtrain_, use_test, use_xtrain):
    # Choose test and train data
    if use_test:
        # Use y_test for test
        if use_xtrain:
            data_xtrain_mapped = map_index_xy(data_ytrain_.index, data_xtrain_)
            data_train = data_ytrain_.join(data_xtrain_mapped)

            data_xtest_mapped = map_index_xy(data_ytest_.index, data_xtest_)
            data_test = data_ytest_.join(data_xtest_mapped)
        else:
            data_train = data_ytrain_
            data_test = data_ytest_
    else:
        # Separate train data in 2 : use 5000 first rows for test
        if use_xtrain:
            data_xtrain_mapped = map_index_xy(data_ytrain_.index, data_xtrain_)
            data_ = data_ytrain_.join(data_xtrain_mapped)
            data_train = data_[5000:]
            data_test = data_[:5000]
        else:
            data_train = data_ytrain[5000:]
            data_test = data_ytrain[:5000]

    return data_train, data_test


def map_index_xy(y_index, df_x):
    # Create new DataFrame by filling missing indexes in df_x
    df_mapped = pd.DataFrame([], index=y_index, columns=df_x.columns)

    x_index = df_x.index
    for index in y_index:
        if index in x_index:
            saved_index = index
        df_mapped.loc[index] = df_x.loc[saved_index]

    return df_mapped


def create_inputs(df, col):
    # Create Dataframe with columns ['month', 'day', 'dayofweek', 'hour', 'minute']
    # from index of df
    # Add columns from df[col] to created DataFrame
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
    y_ = y_.join(df[col])
    return y_



from sklearn.ensemble import RandomForestClassifier

def train_and_predict(train_, target_, test_):
    # train and predict for 1 column
    clf_= RandomForestClassifier(random_state=42, n_estimators= 100)
    clf_.fit(train_, target_)
    pred_ = clf_.predict(test_)
    return pred_


def write_pred(pred_, test_):
    df = pd.DataFrame([], index=test_.index, columns=test_.columns)
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
data_ytrain, data_xtest, df_ytest, data_xtrain, data_static_info = read_data()

# Clean data and map str --> int
df_ytrain = create_df_ytrain(data_ytrain)
df_xtrain = create_df_x(data_xtrain)
df_xtest = create_df_x(data_xtest)

# Use test or train ?
df_train, df_test = define_test(df_ytest, df_ytrain, df_xtest, df_xtrain, use_test=True, use_xtrain=True)

# Change format of input (Timestamp --> int[])
train = create_inputs(df_train, ['average_flow'])
test = create_inputs(df_test, ['average_flow'])

# Predict using random forest
target = df_train[data_ytrain.columns]
predictions = train_and_predict(train, target, test)

# Map back int --> str  and write results in a Dataframe
df_pred = write_pred(predictions, df_ytest)
df_pred.to_csv("y_random.csv")