"""
modified from Vladimir Iglovikov tilii7 and extremin
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

test['loss'] = np.nan
joined = pd.concat([train, test])

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds-labels
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


cat_feature = [n for n in joined.columns if n.startswith('cat')]
cont_feature = [n for n in joined.columns if n.startswith('cont')]


if __name__ == '__main__':


    for column in cat_feature:
        joined[column] = pd.factorize(joined[column].values, sort=True)[0]


    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]

    shift = 200
    y = np.log(train['loss'] + shift)
    ids = test['id']
    X = train.drop(['loss', 'id'], 1)
    X_test = test.drop(['loss', 'id'], 1)

    # X = X.sample(frac=0.1)
    # y = y .iloc[X.index.values]

    n_folds = 5
    kf = KFold(X.shape[0], n_folds=n_folds)
    prediction = np.zeros(ids.shape)
    prediction_oob = np.zeros(train['id'].shape)

    final_fold_prediction = []
    final_fold_real = []

    partial_evalutaion = open('../log/temp_scores.txt', 'w')
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d' % (i + 1))
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        RANDOM_STATE = 2016
        params = {
            'min_child_weight': 1,
            'eta': 0.001,
            'colsample_bytree': 0.5,
            'max_depth': 12,
            'subsample': 0.8,
            'alpha': 1,
            'gamma': 1,
            'silent': 1,
            'verbose_eval': True,
            'seed': RANDOM_STATE
        }

        xgtrain = xgb.DMatrix(X_train, label=y_train)
        xgtrain_2 = xgb.DMatrix(X_val, label=y_val)

        xgtest = xgb.DMatrix(X_test)
        xgx = xgb.DMatrix(X)
        watchlist = [(xgtrain, 'train'), (xgtrain_2, 'eval')]

        model = xgb.train(params, xgtrain, 100000, watchlist, obj=logregobj, feval=evalerror, early_stopping_rounds=300)
        prediction += np.exp(model.predict(xgtest)) - shift
        prediction_oob += np.exp(model.predict(xgx)) - shift

        X_val = xgb.DMatrix(X_val)
        temp_serises = pd.Series(np.exp(model.predict(X_val))-shift)
        final_fold_prediction.append(temp_serises)
        temp_serises = np.exp(y_val) - shift
        final_fold_real.append(temp_serises)

        temp_cv_score = mean_absolute_error(np.exp(model.predict(X_val))-shift, np.exp(y_val) - shift)

        partial_evalutaion.write('fold '+str(i)+' '+str(temp_cv_score)+'\n')
        partial_evalutaion.flush()

    prediction = prediction/n_folds
    prediction_oob = prediction_oob/n_folds
    preds_oob = pd.DataFrame()
    preds_oob['id'] = train['id']
    preds_oob['loss'] = prediction_oob

    preds_oob.to_csv('../result/preds_oob_xgboost.csv', index=False)

    submission = pd.DataFrame()
    submission['id'] = ids
    submission['loss'] = prediction

    submission.to_csv('../result/submission_xgboost_score.csv', index=False)

    final_fold_prediction = pd.concat(final_fold_prediction, ignore_index=True)
    final_fold_real = pd.concat(final_fold_real, ignore_index=True)

    cv_score = mean_absolute_error(final_fold_prediction, final_fold_real)
    print cv_score
# xgb 1110 from Vladimir Iglovikov and tilii7
# https://www.kaggle.com/mtinti/allstate-claims-severity/xgb-1110-from-vladimir-iglovikov-and-tilii7
