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

def neurons_time_PCA(dat,tVar,minT,maxT,toplot=False):
  #Perform PCA with the neurons of a single session. Components are estimated by considering activity in the range minT:maxT
  #Example input neurons_PCA(dat,0.9,51,130)

  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.decomposition import PCA

  NT = dat['spks'].shape[-1]

  #@title top PC directions from stimulus + response period, with projections of the entire duration
  NN = dat['spks'].shape[0]
  #All trials are concatenated to result in a session-long time series for each neuron:
  #droll = np.reshape(dat['spks'][:,:,minT:maxT], (NN,-1)) # first 80 bins = 1.6 sec
  #(N.B. only the time bins for stimulus + response are used!)

  #droll = droll - np.mean(droll, axis=1)[:, np.newaxis] #center each neuron's response vector
  #(np.newaxis is used to add an additional dimension, to be consistent with the original droll matrix)

  droll = reshaping_features(dat['spks']) #[:,:,minT:maxT].mean(axis=2)
  droll = droll.T
  nPCs = min(droll.shape) #set how many PCs we are interested in (all of them)

  model = PCA(n_components = nPCs).fit(droll.T) #perform PCA!
  weights = model.components_ #extract the weight of each PC dimension for each neuron
  PCneurons = weights @ droll #multiply each neuron by its corresponding weights
  #(N.B. the entire trial durations for each neuron are multiplied by the weight! But weight were extracted only from a portion)
  #PCneurons = np.reshape(PCneurons, (nPCs, -1, NT)) #session-long time series are split again into trial-long time series

  explVar = model.explained_variance_
  

  fracVar = explVar/np.sum(explVar) #fraction of explained variance for each PC
  cumVar = np.cumsum(fracVar) #cumulative sum of explained variance
  PCrange = np.max(np.where(cumVar<=tVar)) #The lowest number of components that explain the threshold variance
  
  if toplot == True:
      plt.figure()
      plt.bar(range(nPCs),explVar)
      plt.xlabel('Eigenvector')
      plt.ylabel('Explained variance')
      
      plt.plot([PCrange,PCrange],[0,np.max(explVar)],'r')

      #@title The top PCs capture most variance across the brain. What do they care about?
      plt.figure(figsize= (20, 6))
  
      for iPC in range((PCneurons).shape[0]):

        this_pc = PCneurons[iPC]

        plt.plot(this_pc.mean(axis=0))

        if iPC==0:
          plt.legend(['right only', 'left only', 'neither', 'both'], fontsize=8)
          plt.xlabel('binned time')
          plt.ylabel('Component value')
        # plt.title('PC %d'%j)

  dat['PCs'] = PCneurons
  dat['weights'] = weights
  dat['PCrange'] = PCrange+1

  return dat


