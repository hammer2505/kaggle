import pandas as pd
import numpy as np

################### read data ###################
filename = '../data/train_drop.csv'
traindata = pd.read_csv(filename)
filename1 = '../data/test_drop.csv'
testdata = pd.read_csv(filename1)

data = pd.DataFrame()
pieces = [traindata, testdata]
data = pd.concat(pieces)

data.set_index('id', inplace=True)

categorial_cols = []
for num in range(1, 117):
    cat = 'cat' + str(num)
    categorial_cols.append(cat)

for categorial_col in categorial_cols:
    data[categorial_col] = data[categorial_col].astype('category')

data_cl = data.copy()
for cc in categorial_cols:
    dummies = pd.get_dummies(data_cl[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data_cl.drop(cc, axis=1, inplace=True)
    data_cl = data_cl.join(dummies)
print data_cl.shape

unknown_mask = data['loss'].isnull()
data_test = data_cl[unknown_mask]
data_train = data_cl[unknown_mask]
data_train.to_csv("../data/train_encode.csv")
data_test.to_csv("../data/test_encode.csv")
