import time
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

    # SVM
    def SVM_rbf_regressor(self):
        from sklearn.svm import SVR
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        return model


    def SVM_linear_regressor(self):
        from sklearn.svm import SVR
        model = SVR(kernel='linear', C=1e3)
        return model



    def SVM_poly_regressor(self):
        from sklearn.svm import SVR
        model = SVR(kernel='poly', C=1e3, degree=2)
        return model


# end of class regressor define

m_regressor = regressor()

m_regressors = {
                'RFR': m_regressor.random_forest_regressor(),
                'KNR': m_regressor.K_neighbors_regressor(),  # -1936.87015606 88.2224835863
                'LR': m_regressor.linear_regressor(),  # -1336.47867268 2.80002852333
                'SVMRBF': m_regressor.SVM_rbf_regressor(),
                'SVML': m_regressor.SVM_linear_regressor(),
                'SVMP': m_regressor.SVM_poly_regressor()
               }


def Train(train_data, train_label):
    # num_instances = len(train_data)
    seed = 2016
    processors = -1
    num_folds = 2
    scoring = 'neg_mean_absolute_error'
    kfold = KFold(n_splits=num_folds, random_state=seed)
    model = m_regressors['SVML']
    cv_results = cross_val_score(model, train_data, train_label, cv=kfold, scoring=scoring, n_jobs=processors)
    print cv_results.mean(), cv_results.std()
