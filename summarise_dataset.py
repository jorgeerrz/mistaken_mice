from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from filter import *

def pretty_response_counts(responses):
    (unique, counts) = np.unique(responses, return_counts=True)
    response_labels = np.array(['right (-1)', 'centre (0)', 'left (1)  '])
    count_pct = [100*c/np.sum(counts) for c in counts]
    pretty = ""
    for i, u in enumerate(unique):
        pretty += "{}: {} ({}%)\n" .format( response_labels[int(u)+1], counts[i], int(count_pct[i]) ) 
    return pretty, unique, counts

def count_neurons_by_area(dataset, sessionID):
    session = dataset[sessionID]
    neuron_count = {'brain_area': [], 'n_neurons': []}
    for ba in np.sort(np.unique(session['brain_area'])):
        neuron_count['brain_area'].append(ba)
        ba_filter = session['brain_area']==ba
        n_neurons = session['spks'][ba_filter,:,:].shape[0]
        neuron_count['n_neurons'].append(n_neurons)
    return(neuron_count)

def summarise_session(dataset, sessionID):
    session = dataset[sessionID]
    print("\n\nSession ID: {}\n" .format(sessionID ) )
    
    # brain areas
    neuron_count = count_neurons_by_area(dataset, sessionID)
    for i, ba in enumerate(neuron_count['brain_area']):
        print("{}: {} neurons" .format(ba,neuron_count['n_neurons'][i]))
    
    # responses for all trials
    session_pretty, session_unique, session_counts = pretty_response_counts(session['response'])
    print("\n{} trials\n{}" .format(session['spks'].shape[1], session_pretty ))
    
    # responses for unfair trials
    unfair = filter_spikes(dataset, sessionID,unfair_only=True) 
    unfair_pretty, unfair_unique, unfair_counts = pretty_response_counts(unfair['chcs'])
    print("{} unfair trials\n{}" .format(unfair['spks'].shape[1], unfair_pretty ))

    
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
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    sns.despine(left=True)
    
    sns.distplot(details['n_neurons'], kde = False, color='b', rug=True, ax=axs[0], axlabel ='Number of neurons\nper session')
    axs[0].set_title('{} neurons across sessions'.format(total_neurons))
    sns.distplot(details['n_trials'], kde = False, color='g', rug=True, ax=axs[1], axlabel = 'Number of trials\nper session')
    axs[1].set_title('{} trials across sessions'.format(total_trials))

    return details
