from __future__ import print_function
from __future__ import division
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor as ADaBoost
from sklearn.tree import DecisionTreeRegressor as Tree
from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.metrics import make_scorer

################### my evalerror ###################


def my_evalerror(ground_truth, predictions):
    maee = -(np.abs(np.exp(ground_truth) - np.exp(predictions)).mean())
    return maee

################### read data ###################

train = pd.read_csv('../data/train_encode.csv')
y = train['loss']
X = train.drop(['loss', 'id'], 1)
print (X.isnull().sum().sum())
print (y.isnull().sum().sum())
score = make_scorer(my_evalerror, greater_is_better=True)
print(y.shape)
print(X.shape)
log_file = open("../log/ADaBoostTree-output-from-BOpt.txt", 'a')

################### find best ###################


def adboosttreecv(n_estimators):
    return cross_val_score(ADaBoost(Tree(),
                           n_estimators=int(n_estimators),
                           random_state=2016),
                           X, y, scoring=score,
                           n_jobs=1, cv=3).mean()

if __name__ == "__main__":

    adboosttreeBO = BayesianOptimization(adboosttreecv, {'n_estimators': (0, 1000)})
    print('\n', file=log_file)
    log_file.flush()
    print('Running Bayesian Optimization ...\n')
    adboosttreeBO.maximize(init_points=5, n_iter=20)
    print('-'*53)
    print('Final Results')
    print('ADaBoostTree: %f' % adboosttreeBO.res['max']['max_val'])
    print('\nFinal Results', file=log_file)
    print('ADaBoostTree: %f' % adboosttreeBO.res['max']['max_val'], file=log_file)
    log_file.flush()
    log_file.close()
