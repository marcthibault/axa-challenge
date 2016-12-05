import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import fbeta_score, make_scorer
from utils import *

def rf_create_regr(df_calls):
    max_depth = 5
    n_estimators = 10
    regr_multirf = ExtraTreesRegressor(max_depth=max_depth, n_estimators=n_estimators)
    X,y = create_training_xy_day(df_calls)
    regr_multirf.fit(approx_svd(X,100)[:,:48],y)
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
        print(y_pred.shape)
        score+=np.mean(np.exp(0.1*(y_test-y_pred))-0.1*(y_test-y_pred)-1)
        #score+=np.mean((y_pred-y_test)**2)

    return score/5

def err_fun(y, ypred):
    return np.mean(np.exp(0.1*(y-ypred))-0.1*(y-ypred)-1)

def compute_estimator_rf(X,y):
    tscv = TimeSeriesSplit(n_splits=3)

    regr_rf = RandomForestRegressor()
    pipe = Pipeline([("reduce_dim", PCA()), ("regr", regr_rf)])
    if len(X[0])>48:
        n_features_options = [10, 50, 100, 150]
    else:
        n_features_options = [5, 10, 20, 40]
    n_max_depths = [2, 3, 5]
    n_estimators = [50]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7)],
            'reduce_dim__n_components': n_features_options,
            'regr__max_depth': n_max_depths,
            'regr__n_estimators': n_estimators
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': n_features_options,
            'regr__max_depth': n_max_depths,
            'regr__n_estimators': n_estimators
        },
    ]

    grid = GridSearchCV(pipe, cv=tscv, n_jobs=2, param_grid=param_grid, scoring=make_scorer(err_fun))
    grid.fit(X, y)
    #pd.DataFrame(grid.cv_results_).to_csv("results_gridSearch.csv")

    i_estim_max = np.where(np.array(grid.cv_results_["rank_test_score"]) == 1)[0][0]
    n_estimators_max = grid.cv_results_["param_regr__n_estimators"][i_estim_max]
    max_depth_max = grid.cv_results_["param_regr__max_depth"][i_estim_max]

    try:
        n_features_max = int(grid.cv_results_["param_reduce_dim__k"][i_estim_max])
        pipe = Pipeline([("reduce_dim", SelectKBest(chi2, k=n_features_max)),
                         ("regr", RandomForestRegressor(n_estimators=n_estimators_max, max_depth=max_depth_max))])
    except:
        n_features_max = grid.cv_results_["param_reduce_dim__n_components"][i_estim_max]
        pipe = Pipeline([("reduce_dim", PCA(n_components=n_features_max)),
                         ("regr", RandomForestRegressor(n_estimators=n_estimators_max, max_depth=max_depth_max))])


    print(grid.cv_results_["params"][i_estim_max])
    return pipe

def compute_estimator_gb(X,y):
    tscv = TimeSeriesSplit(n_splits=3)

    pipe = Pipeline([("reduce_dim", PCA()), ("regr", GradientBoostingRegressor())])
    if len(X[0])>48:
        n_features_options = [10, 50, 100, 150]
    else:
        n_features_options = [5, 10, 20, 40]
    n_learning_rates = [0.1,1.0]
    n_max_depths = [2,3,5]
    criterions = ["friedman_mse", "mae", "mse"]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7)],
            'reduce_dim__n_components': n_features_options,
            'regr__max_depth': n_max_depths,
            'regr__learning_rate': n_learning_rates,
            'regr__criterion': criterions
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': n_features_options,
            'regr__max_depth': n_max_depths,
            'regr__learning_rate': n_learning_rates,
            'regr__criterion': criterions
        },
    ]

    grid = GridSearchCV(pipe, cv=tscv, n_jobs=2, param_grid=param_grid, scoring=make_scorer(err_fun))
    grid.fit(X, y)
    #pd.DataFrame(grid.cv_results_).to_csv("results_gridSearch.csv")

    i_estim_max = np.where(np.array(grid.cv_results_["rank_test_score"]) == 1)[0][0]
    criterion_max = grid.cv_results_["param_regr__criterion"][i_estim_max]
    learning_rate_max = grid.cv_results_["param_regr__learning_rate"][i_estim_max]
    max_depth_max = grid.cv_results_["param_regr__max_depth"][i_estim_max]

    try:
        n_features_max = int(grid.cv_results_["param_reduce_dim__k"][i_estim_max])
        pipe = Pipeline([("reduce_dim", SelectKBest(chi2, k=n_features_max)),
                         ("regr", GradientBoostingRegressor(max_depth = max_depth_max,
                                                            learning_rate = learning_rate_max,
                                                            criterion = criterion_max))])
    except:
        n_features_max = grid.cv_results_["param_reduce_dim__n_components"][i_estim_max]
        pipe = Pipeline([("reduce_dim", PCA(n_components=n_features_max)),
                         ("regr", GradientBoostingRegressor(max_depth = max_depth_max,
                                                            learning_rate = learning_rate_max,
                                                            criterion = criterion_max))])


    print(grid.cv_results_["params"][i_estim_max])
    return pipe

def compute_estimator_ab(X,y):
    tscv = TimeSeriesSplit(n_splits=3)

    pipe = Pipeline([("reduce_dim", PCA()), ("regr", AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), loss="exponential"))])
    if len(X[0])>48:
        n_features_options = [10, 50, 100, 150]
    else:
        n_features_options = [5, 10, 20, 40]
    n_learning_rates = [0.1,1.0]
    n_max_depths = [2,3,5]

    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7)],
            'reduce_dim__n_components': n_features_options,
            'regr__learning_rate': n_learning_rates
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': n_features_options,
            'regr__learning_rate': n_learning_rates
        },
    ]

    grid = GridSearchCV(pipe, cv=tscv, n_jobs=2, param_grid=param_grid, scoring=make_scorer(err_fun))
    grid.fit(X, y)
    #pd.DataFrame(grid.cv_results_).to_csv("results_gridSearch.csv")

    i_estim_max = np.where(np.array(grid.cv_results_["rank_test_score"]) == 1)[0][0]
    learning_rate_max = grid.cv_results_["param_regr__learning_rate"][i_estim_max]

    try:
        n_features_max = int(grid.cv_results_["param_reduce_dim__k"][i_estim_max])
        pipe = Pipeline([("reduce_dim", SelectKBest(chi2, k=n_features_max)),
                         ("regr", AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), learning_rate=learning_rate_max))])
    except:
        n_features_max = grid.cv_results_["param_reduce_dim__n_components"][i_estim_max]
        pipe = Pipeline([("reduce_dim", PCA(n_components=n_features_max)),
                         ("regr", AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), loss="exponential", learning_rate=learning_rate_max))])


    print(grid.cv_results_["params"][i_estim_max])
    return pipe


if __name__=="__main__":

    df_data = pd.read_csv("csv/Téléphonie.csv", header=0, sep=";", parse_dates=True, infer_datetime_format=True,
                           index_col=0).sort_index()
    week_matrix = create_week_matrix(df_data, 0)
    day_matrix = create_day_matrix(df_data, 0)

    X_week,y = create_training_xy_hour(week_matrix, 13, 30)
    X_day,y = create_training_xy_hour(day_matrix, 13, 30)

    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)

    score_d_gb = 0
    score_d_ab = 0
    score_d_rf = 0
    score_d_rf = 0
    score_w_gb = 0
    score_w_ab = 0
    score_w_rf = 0
    score_t_gb = 0
    score_t_rf = 0
    score_t = 0
    iter = 0
    for train,test in tscv.split(X_week):
        print("Iter nb: "+str(iter))
        X_w_train, X_w_test, y_train, y_test = X_week[train], X_week[test], y[train], y[test]
        X_d_train, X_d_test = X_day[train], X_day[test]

        x_w_maxs = np.max(X_w_train, axis=1)
        x_d_maxs = np.max(X_d_train, axis=1)



        estim_w_rf = compute_estimator_rf(X_w_train, y_train)
        estim_d_rf = compute_estimator_rf(X_d_train, y_train)
        estim_w_rf_s = compute_estimator_rf(X_w_train/x_w_maxs, y_train/x_w_maxs)
        estim_d_rf_s = compute_estimator_rf(X_d_train/x_d_maxs, y_train/x_d_maxs)
        #estim_w_gb = compute_estimator_gb(X_w_train, y_train)
        #estim_w_ab = compute_estimator_ab(X_w_train, y_train)
        #estim_d_gb = compute_estimator_gb(X_d_train, y_train)
        #estim_d_ab = compute_estimator_ab(X_d_train, y_train)
        estim_w_rf.fit(X_w_train, y_train)
        estim_w_rf_s.fit(X_w_train/x_w_maxs, y_train/x_w_maxs)
        #estim_w_gb.fit(X_w_train, y_train)
        #estim_w_ab.fit(X_w_train, y_train)
        estim_d_rf.fit(X_d_train, y_train)
        estim_d_rf_s.fit(X_d_train/x_d_maxs, y_train/x_d_maxs)
        #estim_d_gb.fit(X_d_train, y_train)
        #estim_d_ab.fit(X_d_train, y_train)

        y_w_pred_rf = np.array(estim_w_rf.predict(X_w_test))
        y_w_pred_rf_s = np.array(estim_w_rf_s.predict(X_w_test/x_w_maxs))
        #y_w_pred_gb = np.array(estim_w_gb.predict(X_w_test))
        #y_w_pred_ab = np.array(estim_w_ab.predict(X_w_test))
        y_d_pred_rf = np.array(estim_d_rf.predict(X_d_test))
        y_d_pred_rf_s = np.array(estim_d_rf_s.predict(X_d_test/x_w_maxs))
        #y_d_pred_gb = np.array(estim_d_gb.predict(X_d_test))
        #y_d_pred_ab = np.array(estim_d_ab.predict(X_d_test))

        iter+=1
        score_w_rf=err_fun(y_test,y_w_pred_rf)
        score_w_rf_s=err_fun(y_test,y_w_pred_rf_s*x_w_maxs)
        #score_w_gb=err_fun(y_test,y_w_pred_gb)
        #score_w_ab=err_fun(y_test,y_w_pred_ab)
        score_d_rf=err_fun(y_test,y_d_pred_rf)
        score_d_rf_s=err_fun(y_test,y_d_pred_rf_s*x_d_maxs)
        #score_d_gb=err_fun(y_test,y_d_pred_gb)
        #score_d_ab=err_fun(y_test,y_d_pred_ab)
        score_t_rf=err_fun(y_test,np.maximum(y_w_pred_rf,y_d_pred_rf))
        #score_t_gb=err_fun(y_test,np.maximum(y_w_pred_gb,y_d_pred_gb))
        #score_t_ab=err_fun(y_test,np.maximum(y_w_pred_ab,y_d_pred_ab))
        #score_t=err_fun(y_test,np.maximum(y_w_pred_gb,np.maximum(y_d_pred_gb,np.maximum(y_w_pred_rf,y_d_pred_rf))))
        #score_t=err_fun(y_test,np.maximum(y_w_pred_ab,np.maximum(y_d_pred_ab,np.maximum(y_w_pred_gb,np.maximum(y_d_pred_gb,np.maximum(y_w_pred_rf,y_d_pred_rf))))))

        print("Current score_w_rf: "+str(score_w_rf))
        print("Current score_w_rf_s: "+str(score_w_rf_s))
        #print("Current score_w_gb: "+str(score_w_gb))
        #print("Current score_w_ab: "+str(score_w_ab))
        print("Current score_d_rf: "+str(score_d_rf))
        print("Current score_d_rf_s: "+str(score_d_rf_s))
        #print("Current score_d_gb: "+str(score_d_gb))
        #print("Current score_d_ab: "+str(score_d_ab))
        print("Current score_t_rf: "+str(score_t_rf))
        #print("Current score_t_gb: "+str(score_t_gb))
        #print("Current score_t_ab: "+str(score_t_ab))
        #print("Current scofe_t: "+str(score_t))
        plt.plot(y_w_pred_rf, label="W_rf", ls="--", color="seagreen")
        plt.plot(y_w_pred_rf_s*x_w_maxs, label="W_rf_s", ls="--", color="coral")
        #plt.plot(y_w_pred_gb, label="W_gb", ls="--", color="coral")
        plt.plot(y_d_pred_rf, label="D_rf", ls=":", color="seagreen")
        plt.plot(y_d_pred_rf_s*x_d_maxs, label="D_rf_s", ls=":", color="coral")
        #plt.plot(y_d_pred_gb, label="D_gb", ls=":", color="coral")
        #plt.plot(y_w_pred_ab, label="W_ab", ls="--", color="royalblue")
        #plt.plot(y_d_pred_ab, label="D_ab", ls=":", color="royalblue")
        plt.plot(y_test, label="T")
        plt.legend()
        plt.show()

