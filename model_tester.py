from summarise_dataset import *
from PCA_fun import neurons_PCA
from GLM import *
from rebin import *
from sklearn.model_selection import *


def test_model(alldat,sessions, brain_areas,pcvar=0.9,unfair_only=True,chosey_only=True,nonzero_only=True,l2_penalty=None, toweight = False,verbose=True):
    output = []
    output = {'Sessions':[],'Accuracies':[],'Biases':[], 'PCcounts':[], 'TrialCounts':[]}
    for session in sessions:
        if np.sum(np.isin(alldat[session]['brain_area'],brain_areas))>0:
            
            filtered = filter_spikes(alldat,session,unfair_only=unfair_only,chosey_only=chosey_only, nonzero_only=nonzero_only,brain_area=brain_areas)
            
            
           
            
            #filtered = rebin(filtered, 50, 250, 2)

            dat = neurons_PCA(filtered,pcvar,0,250)
            
            
            if np.logical_or(np.sum(dat['chcs']==-1)<2,np.sum(dat['chcs']==1)<2):
                continue
                
            dat['PCs'] = dat['PCs'][0:dat['PCrange'],:]
            if l2_penalty:
                penalty = "l2"
                Cpen = l2_penalty
            else:
                penalty = "none"
                Cpen = 1
                
            l1args = {}
            left_bias = np.mean((filtered['chcs'])==1)
            choice_bias = [left_bias if left_bias > .5 else (1.0-left_bias)]
            if toweight:
                w = {-1:left_bias, 1:(1-left_bias)}
                logreg = LogisticRegression(penalty = penalty, C = Cpen,class_weight=w)
            else:
                logreg = LogisticRegression(penalty = penalty, C = Cpen)
           # l1args = {'solver':"saga", 'max_iter':5000}
            splitter = LeaveOneOut()
            splits = splitter.split(dat['PCs'].T)
            accuracies=[]
            predictions = []
            probs = []
            for fit_trials,test_trial in splits:
                logreg.fit((dat['PCs'][:,fit_trials]).T, dat['chcs'][fit_trials])
                prediction = logreg.predict(dat['PCs'][:,test_trial].T)[0]
                prob = logreg.predict_proba(dat['PCs'][:,test_trial].T)[0]
                actual = dat['chcs'][test_trial][0]
                accuracy = int(actual == prediction)
                predictions.append(prediction)
                accuracies.append(accuracy)
                probs.append(prob)
            probs = np.array(probs)
            accuracies = np.array(accuracies)

            aob = accuracies.mean() - choice_bias
            if verbose:
                print("Session# "+ str(session)+" accuracy = "+str(accuracies.mean())+" accuracy over bias = "+str(accuracies.mean() - choice_bias) + ", trial count= "+ str(len(accuracies)),"PC count= "+ str(dat['PCs'].shape[0]))
            output['Sessions'].append(session)
            output['Accuracies'].append(np.mean(accuracies))
            output['Biases'].append(choice_bias)
            output['PCcounts'].append(dat['PCs'].shape[0])
            output['TrialCounts'].append(len(accuracies))

    return output