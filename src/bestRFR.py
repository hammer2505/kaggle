from __future__ import print_function
from __future__ import division
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor as RFR
from bayes_opt import BayesianOptimization
import numpy as np

################### read data ###################

train = pd.read_csv('../data/train_encode.csv')
y = train['loss']
X = train.drop(['loss', 'id'], 1)

print(train.iloc[0, :])
print(y.shape)
print(X.shape)
log_file = open("../log/RandomForest-output-from-BOpt.txt", 'a')
################### find best ###################


def rfrcv(n_estimators, min_samples_split, max_features, max_depth):
    return cross_val_score(RFR(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               max_depth=int(max_depth),
                               random_state=2016,
                               n_jobs=6),
                           X, y, scoring='neg_mean_absolute_error',
                           n_jobs=3, cv=3).mean()

if __name__ == "__main__":

    # rfcBO = BayesianOptimization(rfrcv, {'n_estimators': (10, 500),
    #                                     'min_samples_split': (2, 25),
    #                                     'max_features': (0.1, 0.999),
    #                                     'max_depth': (2, 25)})

    rfcBO = BayesianOptimization(rfrcv, {'n_estimators': (100, 500),
                                         'min_samples_split': (2, 40),
                                         'max_features': (0.1, 0.999),
                                         'max_depth': (25, 40)})

    rfcBO.maximize()
    print('-'*53)
    print('Final Results')
    print('RFC: %f' % rfcBO.res['max']['max_val'])
    print('\nFinal Results', file=log_file)
    print('RandomForest: %f' % rfcBO.res['max']['max_val'], file=log_file)
    log_file.flush()
    log_file.close()

##########################################
# -----------------------------------------------------
# Final Results
# RFC: -1229.487188

#  Step|   Time |      Value |   max_depth |   max_features |   min_samples_split |   n_estimators |
#  26 | 04m29s | -1229.48719 |     25.8703 |         0.3121 |              2.4930 |       307.2629 |
