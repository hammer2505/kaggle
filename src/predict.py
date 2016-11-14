import pandas as pd
import numpy as np
import time
import os
import regression

################### read data ###################
t1 = time.time()

try:
    # notexist
    train = pd.read_csv('../data/train_encode.csv')
    test = pd.read_csv('../data/test_encode.csv')
    y = train['loss']
    X = train.drop(['loss', 'id', 'Unnamed: 0'], 1)
    Xtest = test.drop(['loss', 'id'], 1)
    id_train = train['id'].values
    id_test = test['id'].values
except IOError:
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    test['loss'] = np.nan
    joined = pd.concat([train, test])

    for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)

            def filter_cat(x):
                if x in remove:
                    return np.nan
                    return x

            joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)

        joined[column] = pd.factorize(joined[column].values, sort=True)[0]

    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]

    y = train['loss']
    X = train.drop(['loss', 'id'], 1)
    Xtest = test.drop(['loss', 'id'], 1)
    print X.shape
    train.to_csv('../data/train_encode.csv')
    test.to_csv('../data/test_encode.csv')
    id_train = train['id'].values
    id_test = test['id'].values
else:
    print "data has been loaded!"

del train, test
regression.Result(model='RFR', train_data=X, train_label=y, id_train=id_train,
                  test_data=Xtest, id_test=id_test)
