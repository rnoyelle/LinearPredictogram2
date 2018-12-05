import sys
import os 
import os.path

import numpy as np
#from .matnpyio import get_data
#import matnpyio as io
#import preprocess as pp

from . import matnpyio as io# matnpyio as io
from . import preprocess as pp#import preprocess as pp



def get_preprocessed_from_raw(sess_no, raw_path, align_on, from_time, to_time, lowcut, highcut, order) :
    
    #params
    sess = '01'
       
    trial_length = abs(from_time - to_time)

    # Paths
    #raw_path = base_path + 'data/raw/' + sess_no + '/session' + sess + '/'
    rinfo_path = raw_path + 'recording_info.mat'
    tinfo_path = raw_path + 'trial_info.mat'

    # Define and loop over intervals
    
    srate = io.get_sfreq(rinfo_path) # = 1 000
    n_trials = io.get_number_of_trials(tinfo_path) 
    last_trial = int(max(io.get_trial_ids(raw_path)))
    n_chans = io.get_number_of_channels(rinfo_path)
    channels = [ch for ch in range(n_chans)]

    # Pre-process data
    filtered = np.empty([n_trials,
                        len(channels),
                        int(trial_length * srate/1000)])

    trial_counter = 0; counter = 0
    while trial_counter < last_trial:
        n_zeros = 4-len(str(trial_counter+1))
        trial_str = '0' * n_zeros + str(trial_counter+1)  # fills leading 0s
        if sess == '01' :
            file_in = sess_no + '01.' + trial_str + '.mat'
        else :
            file_in = sess_no + '02.' + trial_str + '.mat'
            
        if align_on == 'sample' :        
            onset = io.get_sample_on(tinfo_path)[trial_counter].item()
        elif align_on == 'match' :
            onset = io.get_match_on(tinfo_path)[trial_counter].item()
        else :
            print("Petit problème avec align_on : 'sample' ou 'match' ")
            

        
        if np.isnan(onset):  # drop trials for which there is no onset info
            print('No onset for ' + file_in)
            trial_counter += 1
            if trial_counter == last_trial:
                break
            else:
                counter += 1
                continue
        print(file_in)
        try:
            raw = io.get_data(raw_path + file_in)
            temp = pp.strip_data(raw,
                                rinfo_path,
                                onset,
                                start=from_time,
                                length=trial_length)
            temp = pp.butter_bandpass_filter(temp,
                                            lowcut,
                                            highcut,
                                            srate,
                                            order)
            if temp.shape[1] == trial_length:  # drop trials shorter than length
                filtered[counter] = temp
            counter += 1
        except IOError:
            print('No file ' + file_in)
        trial_counter += 1

    # Return data

    filtered = np.array(filtered)
    return(filtered)



def get_subset_by_areas(sess_no, raw_path, 
                        align_on, from_time, to_time, lowcut, highcut, target_areas,
                        epsillon=100, order=3,
                        only_correct_trials =True, renorm=True):
    # PATH
    tinfo_path = raw_path + 'trial_info.mat'
    rinfo_path = raw_path + 'recording_info.mat'
    
    # LOAD DATA
    filtered = get_preprocessed_from_raw(sess_no, raw_path, align_on, from_time - epsillon, to_time + epsillon, lowcut, highcut, order)
    

    
    
    # don't keep missing data // keep only_correct_trials if True
    responses = io.get_responses(tinfo_path)
    if only_correct_trials == False:
        ind_to_keep = (responses == responses).flatten()
    else:
        ind_to_keep = (responses == 1).flatten()
        
    filtered = filtered[ind_to_keep, :,:]
    
    # SELECT CHANNELS 
    
    area_names = io.get_area_names(rinfo_path)
    
    idx = []
    for count, area in enumerate(area_names):
        if area in target_areas :
            idx.append(count)
            
    if epsillon !=0 :
        filtered = filtered[:, idx, epsillon : -epsillon ]
    else:
        filtered = filtered[:,idx,:]
        
    # RENORM DATA
        
    if renorm == True :
        filtered = pp.renorm(filtered)
        
        
    return( filtered, area_names[idx] )

