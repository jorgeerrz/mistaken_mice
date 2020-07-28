import numpy as np

#this function will do the filtering
def filter_spikes(alldata, session_id,brain_area=None,unfair_only=True,chosey_only=True,nonzero_only=True):
    #alldata is the collated data from all sessions/neurons/timepoints as shown in the tutorial notebook
    #grab spikes/choices from one section
    dat = alldata[session_id]
    spks = dat['spks']
    chcs = dat['response']
    RTs = dat['response_time']
    ba = dat['brain_area']

    #grab only spikes/choices from trials where left/right contrast is equal and nonzero
    unfair_filter = (dat['contrast_right']==dat['contrast_left'])
    nonzero_filter = np.logical_and((dat['contrast_right'] != 0), (dat['contrast_left'] != 0))
    chosey_filter = dat['response']!=0
    
    
    trial_filter = np.ones(chcs.shape)
    
    if unfair_only:
        trial_filter = np.logical_and(trial_filter,unfair_filter)
    if chosey_only:
        trial_filter = np.logical_and(trial_filter,chosey_filter)
    if nonzero_only:
        trial_filter = np.logical_and(trial_filter,nonzero_filter)

    
    spks = spks[:,trial_filter,:]
    chcs = chcs[trial_filter]
    RTs = RTs[trial_filter]
        

    #grab only spikes from the VISp
    if brain_area:
        spks = spks[np.isin(dat['brain_area'],brain_area),:,:]
        ba = ba[np.isin(dat['brain_area'],brain_area)]

    #grab only spikes from between -500ms and 500ms, relative to stimulus onset (each bin is 10ms)
    #spks = spks[:,:,0:100]

    return {'spks':spks, 'chcs':chcs, 'RTs':RTs, 'brain_area':ba}


def filter_dataset(alldata):
    #alldata is the collated data from all sessions/neurons/timepoints as shown in the tutorial notebook
    #grab spikes/choices from one section
    alldata_f = np.zeros_like(alldata)
    for session_id in range(alldata.shape[0]):
        spikes_f, chcs_f, RTs_f = filter_spikes(alldata,session_id)
        dict_f = {"spks" : spikes_f, "response": chcs_f, "response_time": RTs_f,
        "brain_area": alldata[session_id]['brain_area'], "bin_size": alldata[session_id]['bin_size']}

        alldata_f[session_id] = dict_f

    return alldata_f
