import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def plot_weights(models, sharey=True):
  """Draw a stem plot of weights for each model in models dict."""
  n = len(models)
  f = plt.figure(figsize=(10, 2.5 * n))
  axs = f.subplots(n, sharex=True, sharey=sharey)
  axs = np.atleast_1d(axs)

  for ax, (title, model) in zip(axs, models.items()):
    
    ax.margins(x=.02)
    stem = ax.stem(model.coef_.squeeze(), use_line_collection=True)
    stem[0].set_marker(".")
    stem[0].set_color(".2")
    stem[1].set_linewidths(.5)
    stem[1].set_color(".2")
    stem[2].set_visible(False)
    ax.axhline(0, color="C3", lw=3)
    ax.set(ylabel="Weight", title=title)
  ax.set(xlabel="Feature")
  f.tight_layout()

def compute_accuracy(X, y, model):
  """Compute accuracy of classifier predictions.
  
  Args:
    X (2D array): Data matrix
    y (1D array): Label vector
    model (sklearn estimator): Classifier with trained weights.

  Returns:
    accuracy (float): Proportion of correct predictions.  
  """
  #############################################################################
  # TODO Complete the function, then remove the next line to test it
  #raise NotImplementedError("Implement the compute_accuracy function")
  #############################################################################
  feat_reshaped = reshaping_features(X)
  y_pred = model.predict(feat_reshaped)
  accuracy = 1 - np.mean(y_pred - y) #np.mean(y == y_pred) 

  return accuracy

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
  feat_reshaped = features
  if pen == "l1":
    log_reg = LogisticRegression(penalty=pen, C = lambda_L2, solver="saga", max_iter=5000).fit(feat_reshaped,choices)
  else: 
    log_reg = LogisticRegression(penalty=pen, C = lambda_L2).fit(feat_reshaped,choices)

  models = {
  "Model": log_reg#,
  #"$L_1$ (C = 1)": log_reg_l1,
  }
  plot_weights(models)

  #train_accuracy = compute_accuracy(features, choices, log_reg)
  #print(f"Accuracy on the training data: {train_accuracy:.2%}")

  return log_reg


