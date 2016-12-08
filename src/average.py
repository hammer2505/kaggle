import pandas as pd
import numpy as np
import os
import regression

###############################################################################
# read data
result_keras = pd.read_csv('../result/submission_keras_score.csv')
result_xgboost = pd.read_csv('../result/submission_5fold-average-xgb_fairobj_1130.892212_2016-12-07-13-59.csv')
ids = result_keras['id']
predict_keras = result_keras['loss']
predict_xgboost = result_xgboost['loss']
predict_last = (predict_keras * 3 + predict_xgboost * 7) / 10
predict_last1 = (predict_keras * 4 + predict_xgboost * 6) / 10 #best
predict_last2 = (predict_keras * 5 + predict_xgboost * 5) / 10
predict_last3 = (predict_keras * 6 + predict_xgboost * 4) / 10
predict_last4 = (predict_keras * 7 + predict_xgboost * 3) / 10

submission = pd.DataFrame()
submission['id'] = ids
submission['loss'] = predict_last
submission.to_csv('../result/submission_keras_xgboost_37.csv', index=False)

submission['loss'] = predict_last1
submission.to_csv('../result/submission_keras_xgboost_46.csv', index=False)

submission['loss'] = predict_last2
submission.to_csv('../result/submission_keras_xgboost_55.csv', index=False)

submission['loss'] = predict_last3
submission.to_csv('../result/submission_keras_xgboost_64.csv', index=False)

submission['loss'] = predict_last4
submission.to_csv('../result/submission_keras_xgboost_73.csv', index=False)
