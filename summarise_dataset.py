from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from filter import *
import re

def pretty_response_counts(responses):
    regexpr = re.compile('-*[0-9]')
    if np.shape(responses)[0] == 0:
        unique = np.array([-1,0,1])
        counts = np.array([0,0,0])
        n_trials = 1
    else:
        (unique, counts) = np.unique(responses, return_counts=True)
        n_trials = np.sum(counts)
    response_labels = np.array(['right (-1)', 'centre (0)', 'left (1)  '])
    pretty = ""
    unique_filled = []
    counts_filled = np.zeros(len(response_labels))
    count_pct = np.zeros(len(response_labels))
    for i, resp in enumerate(response_labels):
        resp_n = int(regexpr.findall(resp)[0])
        unique_filled.append(resp_n)
        if resp_n in unique:
            counts_filled[i] = counts[np.where(unique == resp_n)]
        count_pct[i] = 100*counts_filled[i]/n_trials
        pretty += "{}: {} ({}%)\n" .format( resp, int(counts_filled[i]), int(count_pct[i]) ) 
    return pretty, unique_filled, counts_filled, count_pct

def count_neurons_by_area(dataset, sessionID, select_areas = None):
    session = dataset[sessionID]
    neuron_count = {'brain_area': [], 'n_neurons': []}
    if select_areas == None: select_areas = np.sort(np.unique(session['brain_area']))
    for ba in select_areas:
        neuron_count['brain_area'].append(ba)
        ba_filter = session['brain_area']==ba
        n_neurons = session['spks'][ba_filter,:,:].shape[0]
        neuron_count['n_neurons'].append(n_neurons)
    return(neuron_count)

def summarise_session(dataset, sessionID, select_areas = None, unfair_only=True,chosey_only=True,nonzero_only=True):
    session = dataset[sessionID]
    print("\n\nSession ID: {}\n" .format(sessionID ) )
    
    # brain areas
    neuron_count = count_neurons_by_area(dataset, sessionID, select_areas)
    for i, ba in enumerate(neuron_count['brain_area']):
        print("{}: {} neurons" .format(ba,neuron_count['n_neurons'][i]))
    
    # responses for all trials
    session_pretty, session_unique, session_counts, session_pcts = pretty_response_counts(session['response'])
    print("\n{} trials total\n{}" .format(session['spks'].shape[1], session_pretty ))
    
    # responses for filtered trials
    unfair = filter_spikes(dataset, sessionID,select_areas,unfair_only,chosey_only,nonzero_only) 
    unfair_pretty, unfair_unique, unfair_counts, unfair_pcts = pretty_response_counts(unfair['chcs'])
    print("{} non-zero contrast unfair trials with a choice\n{}" .format(unfair['spks'].shape[1], unfair_pretty ))

def plot_session(dataset, sessionID, select_areas = None, unfair_only=True,chosey_only=True,nonzero_only=True):
    session = dataset[sessionID]
    session_pretty, session_unique, session_counts, session_pcts = pretty_response_counts(session['response'])

    unfair = filter_spikes(dataset, sessionID,select_areas,unfair_only,chosey_only,nonzero_only)
    unfair_pretty, unfair_unique, unfair_counts, unfair_pcts = pretty_response_counts(unfair['chcs'])

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})

    neuron_count = count_neurons_by_area(dataset,sessionID, select_areas)
    axs[0].bar(neuron_count['brain_area'], neuron_count['n_neurons'])
    axs[0].set_ylabel("Number of neurons")
    axs[0].set_xlabel("Brain area")

    width = 0.9 
    x = ['all', 'filtered']
    response_labels = ['right {} ({})' .format(int(session_counts[0]), int(unfair_counts[0])), 
                       'centre {} ({})' .format(int(session_counts[1]), int(unfair_counts[1])), 
                       'left {} ({})' .format(int(session_counts[2]), int(unfair_counts[2]))]
    colours = ['b', 'r', 'g']
    for i, u in enumerate(session_unique):
        btm = [np.sum(session_pcts[:int(u)+1]), np.sum(unfair_pcts[:int(u)+1])]
        y = [session_pcts[int(u)+1], unfair_pcts[int(u)+1]]
        plt.bar(x, y, width, bottom = btm, axes = axs[1]) # , color = colours[i]
    
    axs[1].set_xlabel("Trial set")
    axs[1].set_ylabel("Responses")
    axs[0].set_title("Session {}" .format(sessionID))
    plt.legend(response_labels, bbox_to_anchor=(2, 1), loc='upper right')

    
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
