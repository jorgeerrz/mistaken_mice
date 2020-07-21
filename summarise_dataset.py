from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def summarise_dataset(dataset):
	'''
	Summarise reduced dataset 
	INPUT: (filtered) np data array, Steinmetz data
	OUTPUT: plot/table showing no. neurons for each session
	'''
	n_sessions = np.shape(dataset)[0]
	details = np.zeros((n_sessions,3))
	details = {'n_neurons' : np.zeros(n_sessions),
			 'n_trials': np.zeros(n_sessions),
			 'n_timebins': np.zeros(n_sessions)}
	for idx in range(n_sessions):
		this_session = dataset[idx]
		details['n_neurons'][idx], details['n_trials'][idx], details['n_timebins'][idx] = np.shape(this_session['spks'])
	
	total_neurons = int(np.sum(details['n_neurons']))
	total_trials = int(np.sum(details['n_trials']))

	sns.set(style="white", palette="muted", color_codes=True)
	fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharex=True)
	sns.despine(left=True)

	sns.distplot(details['n_neurons'], kde = False, color='b', rug=True, ax=axs[0], axlabel ='Number of neurons\nper session')
	axs[0].set_title('{} neurons across sessions'.format(total_neurons))
	sns.distplot(details['n_trials'], kde = False, color='g', rug=True, ax=axs[1], axlabel = 'Number of trials\nper session')
	axs[1].set_title('{} trials across sessions'.format(total_trials))

	return details

if __name__ == "__main__":

	# Data retrieval
	import os, requests

	fname = []
	for j in range(3):
	  fname.append('steinmetz_part%d.npz'%j)
	url = ["https://osf.io/agvxh/download"]
	url.append("https://osf.io/uv3mw/download")
	url.append("https://osf.io/ehmw2/download")

	for j in range(len(url)):
		if not os.path.isfile(fname[j]):
			try:
				r = requests.get(url[j])
			except requests.ConnectionError:
				print("!!! Failed to download data !!!")
			else:
				if r.status_code != requests.codes.ok:
					print("!!! Failed to download data !!!")
			else:
				with open(fname[j], "wb") as fid:
				fid.write(r.content)
  
	# Data loading
	alldat = np.array([])
	for j in range(len(fname)):
	  alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))
	
	# summarise the dataset
	summarise_dataset(alldat)
