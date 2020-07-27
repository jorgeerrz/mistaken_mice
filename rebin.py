import numpy as np

def rebin(dat, minT, maxT, newbin, oldbin = 0.01):
  ''' takes spike counts from 'spks' and rebins them into a new time bin specified by newbin

  Args:
  input = dict for the session (dat['spks'], dimensions: neurons * trials * time bins)
  newbin = size of the new time bin in seconds
  oldbin = size of old time bin. default 0.01s

  Returns:
  newcount = spike count in new bin
  newbin = size of new bin size
  '''

  #assert newbin > oldbin 'new bin size must be bigger than old bin size'
  input = dat['spks'][:,:,minT:maxT]
  # initialize variables
  poolbin = newbin/oldbin # calculates numbers of bins to be pooled into one new bin
  newcount = np.zeros((input.shape[0],input.shape[1],int(input.shape[2]/poolbin))) # create matrix of zeros with appropriate third dimension (number of bins)


  index = list(range(int(input.shape[2]/poolbin)))
  for idx in index:
      # sum up counts from one step to the next
      slc = input[:,:,int(idx*poolbin):int((idx+1)*poolbin)]
      newcount[:,:,idx] = np.sum(slc, axis=2)
  
  dat2 = dat.copy()
  dat2['spks'] = newcount

  return dat2 #newcount, newbin
