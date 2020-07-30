from summarise_dataset import *
from PCA_fun import neurons_PCA
from GLM import *
from rebin import *
from sklearn.model_selection import *


def test_model(alldat,sessions, brain_areas,pcvar=0.5,unfair_only=True,chosey_only=True,nonzero_only=True,penalty_type='none',Cpen=1, toweight = False,verbose=True):
    output = []
    output = {'Sessions':[],'Accuracies':[],'Biases':[], 'PCcounts':[], 'TrialCounts':[],'Advantages':[],'Brier_GLM':[],'Brier_Bias':[],'Accuracies_bias':[],'Neuroncount':[]}
    for session in sessions:
        if np.sum(np.isin(alldat[session]['brain_area'],brain_areas))>0:
            
            filtered = filter_spikes(alldat,session,unfair_only=unfair_only,chosey_only=chosey_only, nonzero_only=nonzero_only,brain_area=brain_areas)
            
            
           
            
            #filtered = rebin(filtered, 50, 250, 2)

            dat = neurons_PCA(filtered,pcvar,0,250)
            
            
            if np.logical_or(np.sum(dat['chcs']==-1)<2,np.sum(dat['chcs']==1)<2):
                continue
                
            dat['PCs'] = dat['PCs'][0:dat['PCrange'],:]
            
            if penalty_type=='l2':
                penalty = "l2"
                Cpen = Cpen
                solver='liblinear'
                max_iter=5000
            elif penalty_type=='l1':
                penalty = 'l1'
                Cpen = Cpen
                solver="saga"
                max_iter=5000
            elif penalty_type=='none':
                penalty = "none"
                Cpen = 1
                solver='lbfgs'
                max_iter=5000
            else:
                penalty = penalty_type
                Cpen = Cpen
                solver='liblinear'
                max_iter=5000
                
            l1args = {}
            left_bias = np.mean((filtered['chcs'])==1)
            choice_bias = [left_bias if left_bias > .5 else (1.0-left_bias)]
            if toweight:
                w = {-1:left_bias, 1:(1-left_bias)}
                logreg = LogisticRegression(penalty = penalty, C = Cpen,class_weight=w,solver=solver,max_iter=max_iter)
            else:
                logreg = LogisticRegression(penalty = penalty, C = Cpen,solver=solver,max_iter=max_iter)
           # l1args = {'solver':"saga", 'max_iter':5000}
            splitter = LeaveOneOut()
            splits = splitter.split(dat['PCs'].T)
            accuracies=[]
            predictions = []
            probs = []
            training_biases = []
            glm_offsets = []
            bias_offsets = []
            accuracies_bias = []
            for fit_trials,test_trial in splits:
                logreg.fit((dat['PCs'][:,fit_trials]).T, dat['chcs'][fit_trials])
                prediction = logreg.predict(dat['PCs'][:,test_trial].T)[0]
                prob = logreg.predict_proba(dat['PCs'][:,test_trial].T)[0]
                prob_left = prob[1]
                actual = dat['chcs'][test_trial][0]
                accuracy = int(actual == prediction)
                predictions.append(prediction)
                accuracies.append(accuracy)
                probs.append(prob)
                training_bias_left_i = np.mean((filtered['chcs'][fit_trials])==1)
                trainng_choice_bias_i = [training_bias_left_i if training_bias_left_i > .5 else (1.0-training_bias_left_i)]
                training_biases.append(trainng_choice_bias_i)
                
                if training_bias_left_i>.5:
                    bias_prediction = 1
                elif training_bias_left_i<.5:
                    bias_prediction = -1
                else:
                    bias_prediction = np.random.choice([-1,1])
 
                accuracy_bias = .5 if bias_prediction==.5 else int(actual == bias_prediction)
                
                accuracies_bias.append(accuracy_bias)
                
                glm_offsets.append(prob_left-((actual+1)/2))
                bias_offsets.append(training_bias_left_i-((actual+1)/2))
                
            probs = np.array(probs)
            accuracies = np.array(accuracies)
            accuracies_bias = np.array(accuracies_bias)
            training_biases = np.array(training_biases)
            aob = accuracies.mean() - choice_bias
            
            
            advantage = np.mean(np.square(bias_offsets)-np.square(glm_offsets))
            
            if verbose:
                print("Session# "+ str(session)+" accuracy = "+str(accuracies.mean())+" bias= "+str(choice_bias)+", trial count= "+ str(len(accuracies)),"PC count= "+ str(dat['PCs'].shape[0]),"Bias accuracy:"+str(np.mean(accuracies_bias)))
            output['Sessions'].append(session)
            output['Accuracies'].append(np.mean(accuracies))
            output['Accuracies_bias'].append(np.mean(accuracies_bias))
            output['Advantages'].append(advantage)
            output['Biases'].append(training_biases)
            
            output['Brier_GLM'].append(np.mean(np.square(glm_offsets)))
            output['Brier_Bias'].append(np.mean(np.square(bias_offsets)))
            output['Neuroncount'].append(filtered['spks'].shape[0])
            output['PCcounts'].append(dat['PCs'].shape[0])
            output['TrialCounts'].append(len(accuracies))

    return output