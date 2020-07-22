# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:36:53 2020

@author: Iacopo
"""


def neurons_PCA(dat,tVar,minT,maxT):
    #Perform PCA with the neurons of a single session. Components are estimated by considering activity in the range minT:maxT
    #Example input neurons_PCA(dat,0.9,51,130)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    NT = dat['spks'].shape[-1]
    
    #@title top PC directions from stimulus + response period, with projections of the entire duration    
    NN = dat['spks'].shape[0]
    #All trials are concatenated to result in a session-long time series for each neuron:
    droll = np.reshape(dat['spks'][:,:,minT:maxT], (NN,-1)) # first 80 bins = 1.6 sec
    #(N.B. only the time bins for stimulus + response are used!)    
    
    droll = droll - np.mean(droll, axis=1)[:, np.newaxis] #center each neuron's response vector
    #(np.newaxis is used to add an additional dimension, to be consistent with the original droll matrix)

    nPCs = NN #set how many PCs we are interested in (all of them)
    model = PCA(n_components = nPCs).fit(droll.T) #perform PCA!
    weights = model.components_ #extract the weight of each PC dimension for each neuron
    PCneurons = weights @ np.reshape(dat['spks'], (NN,-1)) #multiply each neuron by its corresponding weights
    #(N.B. the entire trial durations for each neuron are multiplied by the weight! But weight were extracted only from a portion)
    PCneurons = np.reshape(PCneurons, (nPCs, -1, NT)) #session-long time series are split again into trial-long time series

    explVar = model.explained_variance_
    plt.figure()
    plt.bar(range(nPCs),explVar)
    plt.xlabel('Eigenvector')
    plt.ylabel('Explained variance')
    
    tVar = 0.9
    fracVar = explVar/np.sum(explVar) #fraction of explained variance for each PC
    cumVar = np.cumsum(fracVar) #cumulative sum of explained variance
    PCrange = np.max(np.where(cumVar<=tVar)) #The lowest number of components that explain the threshold variance
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
    dat['PCrange'] = PCrange

    return dat
