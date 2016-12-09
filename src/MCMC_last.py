import pandas as pd
import numpy as np
import os
import regression
from pymc3 import *
from sklearn.metrics import mean_absolute_error
import time

filename1 = '../result/preds_oob_keras_score_sort.csv'
df1 = pd.read_csv(filename1)
model1 = df1['loss']

filename2 = '../result/preds_oob_LR.csv'
df2 = pd.read_csv(filename2)
model2 = df2['loss']

filename3 = '../data/train.csv'
df3 = pd.read_csv(filename3)
y = df3['loss']
print "train start>>>"
data = dict(x1=model1, x2=model2, y=y)
time1 = time.time()
with Model() as model:
    # specify glm and pass in data. The resulting linear model, its likelihood
    # and all its parameters are automatically added to our model.
    glm.glm('y ~ x1 + x2+0', data)
    step = NUTS()  # Instantiate MCMC sampling algorithm
    trace = sample(30000, step, progressbar=True)
time2 = time.time()
print "train end"
print "time use:", (time2 - time1) / 60, "min"
#intercept = np.median(trace.Intercept)
#print(intercept)
x1param = np.median(trace.x1)
print(x1param)
x2param = np.median(trace.x2)
print(x2param)

result1 = '../result/submission_keras_score.csv'
result1 = pd.read_csv(result1)
result1 = result1['loss']

result2 = '../result/submission_LR.csv'
result2 = pd.read_csv(result2)
result2 = result2['loss']

#result = intercept + result1 * x1param + result2 * x2param
result = result1 * x1param + result2 * x2param
filename4 = '../data/sample_submission.csv'
df4 = pd.read_csv(filename4)
df4['loss'] = result
df4.to_csv('../result/MCMC_LR_keras.csv', index=False)
