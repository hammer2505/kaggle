import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack

################### read data ###################
train = pd.read_csv('../data/train_drop.csv')
test = pd.read_csv('../data/train_drop.csv')

# set test loss to NaN
test['loss'] = np.nan

# response and IDs
y = train['loss'].values
id_train = train['id'].values
id_test = test['id'].values

# stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis=0)

# Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

# sparse train and test data
xtr_te = hstack(sparse_data, format='csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

data_train.to_csv("../data/train_encode.csv")
data_test.to_csv("../data/test_encode.csv")
