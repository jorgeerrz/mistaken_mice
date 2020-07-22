import numpy as np

#this function will do the filtering
def filter_spikes(alldata, session_id):
    #alldata is the collated data from all sessions/neurons/timepoints as shown in the tutorial notebook
    #grab spikes/choices from one section
    dat = alldata[session_id]
    spks = dat['spks']
    chcs = dat['response']
    RTs = dat['response_time']

    #grab only spikes/choices from trials where left/right contrast is equal and nonzero
    unfair_filter = np.logical_and((dat['contrast_right']==dat['contrast_left']), (dat['contrast_right'] != 0))
    unfair_chosey_filter = np.logical_and(unfair_filter,(dat['response']!=0))
    spks = spks[:,unfair_chosey_filter,:]
    chcs = chcs[unfair_chosey_filter]
    RTs = RTs[unfair_chosey_filter]

    #grab only spikes from the VISp
    spks = spks[dat['brain_area']=='VISp',:,:]

    #grab only spikes from between -500ms and 500ms, relative to stimulus onset (each bin is 10ms)
    spks = spks[:,:,0:100]

    return spks, chcs, RTs

def filter_dataset(alldata):
    #alldata is the collated data from all sessions/neurons/timepoints as shown in the tutorial notebook
    #grab spikes/choices from one section
    alldata_f = np.zeros_like(alldata)
    for session_id in range(alldata.shape):       
        spikes_f, chcs_f, RTs_f = filter_spikes(alldata,session_id)
        dict_f = {"spks" : spikes_f, "response": chcs_f, "response_time": RTs_f, 
        "brain_area": alldata[session_id]['brain_area'], "bin_size": alldata[session_id]['bin_size']}

        alldata_f[session_id] = dict_f

    return alldata_f
