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

# end of class regressor define

m_regressor = regressor()

m_regressors = {
                 'RFR': m_regressor.random_forest_regressor()
                 }


def Train(train_data, train_label):
    # num_instances = len(train_data)
    seed = 2016
    processors = -1
    num_folds = 2
    scoring = 'neg_mean_absolute_error'
    kfold = KFold(n_splits=num_folds, random_state=seed)
    model = m_regressors['RFR']
    cv_results = cross_val_score(model, train_data, train_label, cv=kfold, scoring=scoring, n_jobs=processors)
    print cv_results.mean(), cv_results.std()
