from __future__ import print_function
from __future__ import division
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor as RFR
from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.metrics import make_scorer

################### my evalerror ###################


def my_evalerror(ground_truth, predictions):
    maee = np.abs(np.exp(ground_truth) - np.exp(predictions)).mean()
    return maee

################### read data ###################

train = pd.read_csv('../data/train_encode.csv')
X = train.drop(['loss', 'id'], 1)
shift = 200
y = np.log(train['loss'] + shift)

print(y.shape)
print(X.shape)
score = make_scorer(my_evalerror, greater_is_better=True)

################### find best ###################


def rfrcv(n_estimators, min_samples_split, max_features, max_depth):
    return cross_val_score(RFR(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               max_depth=int(max_depth),
                               random_state=2016,
                               n_jobs=6),
                           X, y, scoring=score,
                           n_jobs=3, cv=3).mean()


if __name__ == "__main__":

    rfcBO = BayesianOptimization(rfrcv, {'n_estimators': (10, 500),
                                         'min_samples_split': (2, 25),
                                         'max_features': (0.1, 0.999),
                                         'max_depth': (2, 25)})

    rfcBO.maximize()
    print('-'*53)
    print('Final Results')
    print('RFC: %f' % rfcBO.res['max']['max_val'])
