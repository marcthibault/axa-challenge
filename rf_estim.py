import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from utils import *

def create_xy(df_calls):
    X = np.array([])
    Y = np.array([])

    for w in range(0,len(df_calls)-2):
        if np.sum(np.isnan(df_calls[w]))==0 and np.sum(np.isnan(df_calls[w+2,:48]))==0:
            if len(X)==0:
                X = np.array([df_calls[w]])
                Y = np.array([df_calls[w+2,:48]])
            else:
                X = np.append(X, [df_calls[w]], axis=0)
                Y = np.append(Y, [df_calls[w+2,:48]], axis=0)

    return X,Y

if __name__=="__main__":

    df_data = pd.read_csv("csv/CMS.csv", header=0, sep=";", parse_dates=True, infer_datetime_format=True,
                           index_col=0).sort_index()

    X, Y = create_xy(create_matrix(df_data, 6))

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.8)

    max_depth = 20
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                                              random_state=0))
    regr_multirf.fit(X_train, y_train)

    # Predict on new data
    y_multirf = regr_multirf.predict(X_test)

    # Plot the results
    plt.figure()
    s = 50
    a = 0.4
    for i in range(10,14):
        plt.plot(y_test[i], label="True_"+str(i))
        plt.plot(np.ceil(y_multirf[i]), label="Estim_"+str(i), ls="--", linewidth=2)

    plt.title("Multi-output meta estimator")
    plt.legend()
    plt.show()
