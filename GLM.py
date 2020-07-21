import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def reshaping_features(features, axis_pc = 0, axis_trial = 1, axis_time = 2):
  """Function to reshape N x trials x time into a single array, N is number of neurons or PCs
  Output: a matrix with shape trials x (N+time)
  the dimension N + time was concatenated by time, i.e. the first N entries correspond to the N features at time = 0, and so on.
  """
  out = np.zeros((features.shape[axis_trial],features.shape[axis_pc]*features.shape[axis_time]))
  for i in range(features.shape[axis_trial]):
    out[i,:] = features[:,i,:].T.flatten()
  return out

def GLM_logistic(features, choices, pen = "l2", lambda_L2 = 1):
  """Logistic regression already implemented in sci-kit learn
  	 Input: features are in format features x trials x time
  	 Output: the trained model, weights
  """
  feat_reshaped = reshaping_features(features)
  if pen == "l1":
    log_reg = LogisticRegression(penalty=pen, C = lambda_L2, solver="saga", max_iter=5000).fit(feat_reshaped,choices)
  else: 
    log_reg = LogisticRegression(penalty=pen, C = lambda_L2).fit(feat_reshaped,choices)
  return log_reg