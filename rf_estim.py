import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from utils import *

def rf_create_regr(df_calls):
    max_depth = 5
    n_estimators = 10
    regr_multirf = ExtraTreesRegressor(max_depth=max_depth, n_estimators=n_estimators)
    X,y = create_xy(df_calls)
    regr_multirf.fit(approx_svd(X,100),y)
    return regr_multirf

def score_estim(X, y, estim):
    score=0.0
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estim.fit(approx_svd(X_train,10),y_train)
        y_pred = estim.predict(X_test)
        print(y_train.shape)
        print(y_pred.shape)
        score+=np.mean(np.exp(0.1*(y_test-y_pred))-0.1*(y_test-y_pred)-1)
        #score+=np.mean((y_pred-y_test)**2)

    return score/5



if __name__=="__main__":

    df_data = pd.read_csv("csv/Services.csv", header=0, sep=";", parse_dates=True, infer_datetime_format=True,
                           index_col=0).sort_index()

    X, Y = create_xy(create_matrix(df_data, 0))

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.4)

    max_depth = 2
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                                              random_state=0, n_estimators=50))
    regr_rf = RandomForestRegressor(max_depth=max_depth, n_estimators=200)
    print("Score RF: "+str(float(score_estim(X,Y,regr_rf))))
    regr_rf_extra = ExtraTreesRegressor(max_depth=max_depth, n_estimators=200)
    print("Score EF: " + str(float(score_estim(X, Y, regr_rf_extra))))
    regr_adab = MultiOutputRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth), n_estimators=200))
    print("Score Adab: " + str(float(score_estim(X, Y, regr_adab))))



    #regr_multirf.fit(X_train, y_train)
    # regr_rf.fit(approx_svd(X_train,150), y_train)
    # regr_adab.fit(approx_svd(X_train,150), y_train)
    # regr_rf_extra.fit(approx_svd(X_train,150), y_train)
    # #regr_adab.fit(X_train, y_train)
    # original_params = {'n_estimators': 1000, 'max_leaf_nodes': 8, 'max_depth': None,
    #                  'min_samples_split': 5}
    # regr_gb = MultiOutputRegressor(GradientBoostingRegressor(**original_params, learning_rate=0.1))
    # regr_gb.fit(X_train, y_train)
    # Plot the results
    #y_multirf = regr_multirf.predict(X_test)
    # y_rf = regr_rf.predict(X_test)
    # y_adab = regr_adab.predict(X_test)
    # y_rf_extra = regr_rf_extra.predict(X_test)
    # # y_gb = regr_gb.predict(X_test)
    #
    # plt.figure()
    #
    # for i in range(10,12):
    #     plt.plot(y_test[i], label="True_"+str(i))
    #     plt.plot(np.ceil(y_rf_extra[i]), label="Estim_extra_"+str(i)+"_score_"+str(regr_rf_extra.score(X_test, y_test)), ls="--", linewidth=2)
    #     plt.plot(np.ceil(y_rf[i]), label="Estim_rf_"+str(i)+"_score_"+str(regr_rf.score(X_test, y_test)), ls="-.", linewidth=2)
    #     plt.plot(np.ceil(y_adab[i]), label="Estim_adab_"+str(i)+"_score_"+str(regr_adab.score(X_test, y_test)), ls=":", linewidth=2)
    #
    # plt.title("Multi-output meta estimator")
    # plt.legend()
    # plt.show()


    # feature_importance = regr_rf.feature_importances_
    # # make importances relative to max importance
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # sorted_idx = np.argsort(feature_importance)
    #
    # pos = np.arange(sorted_idx.shape[0]) + .5
    # plt.barh(pos, feature_importance[sorted_idx], align='center', alpha=0.5, color="seagreen")
    # plt.yticks(pos, sorted_idx)
    # plt.xlabel('Relative Importance')
    # plt.title('Variable Importance')
    # plt.show()