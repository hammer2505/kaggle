import pandas as pd
import numpy as np
import os
import regression

###############################################################################
# read data
result_keras = pd.read_csv('../result/submission_keras_score.csv')
result_xgboost = pd.read_csv('../result/submission_xgboost.csv')
ids = result_keras['id']
predict_keras = result_keras['loss']
predict_xgboost = result_xgboost['loss']
predict_last = (predict_keras * 6 + predict_xgboost * 4) / 10
predict_last1 = (predict_keras * 7 + predict_xgboost * 3) / 10

submission = pd.DataFrame()
submission['id'] = ids
submission['loss'] = predict_last
submission.to_csv('../result/submission_keras_xgboost_64.csv', index=False)

submission['loss'] = predict_last1
submission.to_csv('../result/submission_keras_xgboost_73.csv', index=False)
