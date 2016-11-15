import time
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


class regressor:

    # Random Forest Regressor
    def random_forest_regressor(self):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
        return model

    # K Neighbors Regressor
    def K_neighbors_regressor(self):
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor()
        return model

    # Linear Regressor
    def linear_regressor(self):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        return model

    # SVM_rbf
    def SVM_rbf_regressor(self):
        from sklearn.svm import SVR
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        return model

    # SVM_linear
    def SVM_linear_regressor(self):
        from sklearn.svm import SVR
        model = SVR(kernel='linear', C=1e3)
        return model

    # SVM_poly
    def SVM_poly_regressor(self):
        from sklearn.svm import SVR
        model = SVR(kernel='poly', C=1e3, degree=2)
        return model

    # Adaboost_LR
    def ADaboost_Tree_regressor(self):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        model = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(),
                                  n_estimators=100,
                                  random_state=2016)
        return model


# end of class regressor define

def my_evalerror(ground_truth, predictions):
    maee = -(np.abs(np.exp(ground_truth) - np.exp(predictions)).mean())
    return maee

m_regressor = regressor()

m_regressors = {
                'RFR': m_regressor.random_forest_regressor(),
                'KNR': m_regressor.K_neighbors_regressor(),  # -1936.87015606 88.2224835863
                'LR': m_regressor.linear_regressor(),  # -1336.47867268 2.80002852333
                'SVMRBF': m_regressor.SVM_rbf_regressor(),
                'SVML': m_regressor.SVM_linear_regressor(),
                'SVMP': m_regressor.SVM_poly_regressor(),
                'ADaTree': m_regressor.ADaboost_Tree_regressor()
               }


def Train(train_data, train_label):
    # num_instances = len(train_data)
    seed = 2016
    processors = -1
    num_folds = 2
    # scoring = 'neg_mean_absolute_error'
    kfold = KFold(n_splits=num_folds, random_state=seed)
    model = m_regressors['ADaTree']
    score = make_scorer(my_evalerror, greater_is_better=True)
    cv_results = cross_val_score(model, train_data, train_label, cv=kfold, scoring=score, n_jobs=processors)
    print cv_results.mean(), cv_results.std()


def Result(model, train_data, train_label, id_train, test_data, id_test):
    from sklearn.ensemble import BaggingRegressor
    seed = 2016
    processors = -1
    num_folds = 2
    regressor = BaggingRegressor(base_estimator=m_regressors[model], n_jobs=-1)
    regressor.fit(train_data, train_label)

    pred_oob = regressor.predict(train_data)
    print model
    print '################ score ##############'
    print my_evalerror(train_label, pred_oob)
    df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
    df.to_csv('../result/preds_oob_'+model+'.csv', index=False)

    pred_test = regressor.predict(test_data)
    df = pd.DataFrame({'id': id_test, 'loss': pred_test})
    df.to_csv('../result/submission_'+model+'.csv', index=False)
