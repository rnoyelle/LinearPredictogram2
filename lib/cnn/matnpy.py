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

def get_preprocessed_from_raw2(sess_no, raw_path, 
                              align_on1, from_time1, to_time1, lowcut1, highcut1, order1, 
                              align_on2, from_time2, to_time2, lowcut2, highcut2, order2) :
    
    #params
    sess = '01'
       
    trial_length1 = abs(from_time1 - to_time1)
    trial_length2 = abs(from_time2 - to_time2)

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
    filtered1 = np.empty([n_trials,
                        len(channels),
                        int(trial_length1 * srate/1000)])
    filtered2 = np.empty([n_trials,
                        len(channels),
                        int(trial_length2 * srate/1000)])

    trial_counter = 0; counter = 0
    while trial_counter < last_trial:
        n_zeros = 4-len(str(trial_counter+1))
        trial_str = '0' * n_zeros + str(trial_counter+1)  # fills leading 0s
        if sess == '01' :
            file_in = sess_no + '01.' + trial_str + '.mat'
        else :
            file_in = sess_no + '02.' + trial_str + '.mat'
            
        if align_on1 == 'sample' :        
            onset1 = io.get_sample_on(tinfo_path)[trial_counter].item()
        elif align_on1 == 'match' :
            onset1 = io.get_match_on(tinfo_path)[trial_counter].item()
        else :
            print("Petit problème avec align_on : 'sample' ou 'match' ")
            
        if align_on2 == 'sample' :        
            onset2 = io.get_sample_on(tinfo_path)[trial_counter].item()
        elif align_on1 == 'match' :
            onset2 = io.get_match_on(tinfo_path)[trial_counter].item()
        else :
            print("Petit problème avec align_on : 'sample' ou 'match' ")
            

        
        if np.isnan(onset1) or np.isnan(onset2):  # drop trials for which there is no onset info
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
            temp1 = pp.strip_data(raw,
                                rinfo_path,
                                onset1,
                                start=from_time1,
                                length=trial_length1)
            temp1 = pp.butter_bandpass_filter(temp1,
                                            lowcut1,
                                            highcut1,
                                            srate,
                                            order1)
            temp2 = pp.strip_data(raw,
                                rinfo_path,
                                onset2,
                                start=from_time2,
                                length=trial_length2)
            temp2 = pp.butter_bandpass_filter(temp2,
                                            lowcut2,
                                            highcut2,
                                            srate,
                                            order2)
            
            if temp1.shape[1] == trial_length1:  # drop trials shorter than length
                filtered1[counter] = temp1
            if temp2.shape[1] == trial_length2:  # drop trials shorter than length
                filtered2[counter] = temp2
                
            counter += 1
        except IOError:
            print('No file ' + file_in)
        trial_counter += 1

    # Return data

    filtered1 = np.array(filtered1)
    filtered2 = np.array(filtered2)
    return(filtered1, filtered2)


def get_subset_by_cortex(sess_no, raw_path, 
               align_on1, from_time1, to_time1, window_size1, lowcut1, highcut1, cortex1,
               align_on2, from_time2, to_time2, window_size2, lowcut2, highcut2, cortex2,
               step, delay,
               epsillon = 10, order = 3,
               only_correct_trials = True, renorm = True ):

    tinfo_path = raw_path + 'trial_info.mat'
    rinfo_path = raw_path + 'recording_info.mat'
    
    # get all data
    data1_filtered, data2_filtered = get_preprocessed_from_raw2(sess_no, raw_path, 
                                              align_on1, from_time1 - epsillon, to_time1 + epsillon, lowcut1, highcut1, order, 
                                              align_on2, from_time2 - epsillon, to_time2 + epsillon, lowcut2, highcut2, order)



    #data = get_preprocessed_from_raw(sess_no, raw_path, align_on, from_time, to_time, lowcut, highcut, order)
    
    # don't keep missing data // keep only_correct_trials if True
    
    responses = io.get_responses(tinfo_path)
    if only_correct_trials == False:
        ind_to_keep = (responses == responses).flatten()
    else:
        ind_to_keep = (responses == 1).flatten()
        
    #data1 =data1[ind_to_keep, :, :] # in the same time
    #data2 =data2[ind_to_keep, :, :]
    
    data1_filtered = data1_filtered[ind_to_keep,:,:]
    data2_filtered = data2_filtered[ind_to_keep,:,:]
    
    # select electrode
    
    dico_area_to_cortex = io.get_dico_area_to_cortex()
    area_names = io.get_area_names(rinfo_path)
    
    dtype = [('name', '<U6'), ('index', int), ('cortex', '<U16')]
    values = []
    for count, area in enumerate(area_names):
        if area in dico_area_to_cortex: # if not, area isn't in Visual or Parietal or Prefontal or Motor or Somatosensory
            
            values.append( (area, count, dico_area_to_cortex[area])  )
        else:
            print('Unknow area')
                    
    s = np.array(values, dtype=dtype)
    
    elec1 = s[s['cortex'] == cortex1]['index']
    elec2 = s[s['cortex'] == cortex2]['index']
    
    

    i_max = min ( int(data1_filtered.shape[2] - 2*epsillon - window_size1)//step, 
                int(data2_filtered.shape[2] - 2*epsillon - window_size2)//step ) +1
    ### Get list of step
    list_step = []
    for i in range(i_max):
        list_step = list_step + data1_filtered.shape[0] * [i *step]
    time_step = np.array(list_step)
    # get list of 
    classes = 5
    targets = io.get_samples(tinfo_path)
    targets = targets[ind_to_keep].astype(int).flatten()
    targets =  np.tile(targets, i_max)
    stim = np.eye(classes)[targets].reshape(targets.shape[0], classes)
        
    # concat the data of every time step
    for i in range(i_max):
        if i ==0 :
            data1 = data1_filtered[:, 
                        elec1, 
                        epsillon + i*step : epsillon + i*step + window_size1 ]
            
            data2 = data2_filtered[:, 
                        elec2, 
                        epsillon + i*step : epsillon + i*step + window_size2 ]
            
        else:
            data1 = np.concatenate( (data1, 
                                    data1_filtered[:, 
                                        elec1,
                                        epsillon + i*step : epsillon + i*step + window_size1 ]),
                                    axis =0 )
            data2 = np.concatenate( (data2, 
                                    data2_filtered[:, 
                                        elec2, 
                                        epsillon + i*step : epsillon + i*step + window_size2 ]) ,
                                    axis =0 )
                                    
    
    ### variable for shape
    #n_chans1 = len(elec1)
    #n_chans2 = len(elec2)
            
    #samples_per_trial1 = data1.shape[2] # = window_size1
    #samples_per_trial2 = data2.shape[2] # = window_size2
    
    # renorm data : mean = 0 and var = 1
    if renorm == True :
        data1 = pp.renorm(data1)
        data2 = pp.renorm(data2)

    ## change type and reshape
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)

    data1 = np.reshape(data1, ( data1.shape[0], data1.shape[1], data1.shape[2], 1 ) )
    data2 = np.reshape(data2, ( data2.shape[0], data2.shape[1], data2.shape[2], 1 ) )
    
    return( data1, data2, area_names[elec1], area_names[elec2], time_step, stim )



def get_subset_by_areas(sess_no, raw_path, 
               align_on1, from_time1, to_time1, window_size1, lowcut1, highcut1, target_areas1,
               align_on2, from_time2, to_time2, window_size2, lowcut2, highcut2, target_areas2,
               step, delay,
               epsillon = 100, order = 3,
               only_correct_trials = True, renorm = True ):

    tinfo_path = raw_path + 'trial_info.mat'
    rinfo_path = raw_path + 'recording_info.mat'
    
    # get all data
    data1_filtered, data2_filtered = get_preprocessed_from_raw2(sess_no, raw_path, 
                                              align_on1, from_time1 - epsillon, to_time1 + epsillon, lowcut1, highcut1, order, 
                                              align_on2, from_time2 - epsillon, to_time2 + epsillon, lowcut2, highcut2, order)



    #data = get_preprocessed_from_raw(sess_no, raw_path, align_on, from_time, to_time, lowcut, highcut, order)
    
    # don't keep missing data // keep only_correct_trials if True
    
    responses = io.get_responses(tinfo_path)
    if only_correct_trials == False:
        ind_to_keep = (responses == responses).flatten()
    else:
        ind_to_keep = (responses == 1).flatten()
        
    #data1 =data1[ind_to_keep, :, :] # in the same time
    #data2 =data2[ind_to_keep, :, :]
    
    data1_filtered = data1_filtered[ind_to_keep,:,:]
    data2_filtered = data2_filtered[ind_to_keep,:,:]
    
    # select electrode
    area_names = io.get_area_names(rinfo_path)
    
    idx1 = []
    idx2 = []
    for count, area in enumerate(area_names):
        if area in target_areas1 :
            idx1.append(count)
        if area in target_areas2 :
            idx2.append(count)
            
    if epsillon != 0 :    
        data1_filtered = data1_filtered[:, idx1, epsillon : -epsillon ]
        data2_filtered = data2_filtered[:, idx2, epsillon : -epsillon ]
    else:
        data1_filtered = data1_filtered[:, idx1, :]
        data2_filtered = data2_filtered[:, idx2, :]
        

    i_max = min ( int(data1_filtered.shape[2]  - window_size1)//step, 
                int(data2_filtered.shape[2] - window_size2)//step ) +1
    ### Get list of step
    list_step = []
    for i in range(i_max):
        list_step = list_step + data1_filtered.shape[0] * [i *step]
    time_step = np.array(list_step)
    
    # get list of stim
    classes = 5
    targets = io.get_samples(tinfo_path)
    targets = targets[ind_to_keep].astype(int).flatten()
    targets =  np.tile(targets, i_max)
    stim = np.eye(classes)[targets].reshape(targets.shape[0], classes)
    
    # concat data with time step
    for i in range(i_max):
        if i ==0 :
            data1 = data1_filtered[:, 
                        :, 
                         + i*step : i*step + window_size1 ]
            
            data2 = data2_filtered[:, 
                        :, 
                        i*step : i*step + window_size2 ]
        else:
            data1 = np.concatenate( (data1, 
                                    data1_filtered[:, 
                                        :,
                                        i*step : i*step + window_size1 ]),
                                    axis =0 )
            data2 = np.concatenate( (data2, 
                                    data2_filtered[:, 
                                        :, 
                                        i*step : i*step + window_size2 ]) ,
                                    axis =0 )
                                    
    
    ### variable for shape
    #n_chans1 = len(elec1)
    #n_chans2 = len(elec2)
            
    #samples_per_trial1 = data1.shape[2] # = window_size1
    #samples_per_trial2 = data2.shape[2] # = window_size2
    
    # renorm data : mean = 0 and var = 1
    if renorm == True :
        data1 = pp.renorm(data1)
        data2 = pp.renorm(data2)

    ## change type and reshape
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)

    data1 = np.reshape(data1, ( data1.shape[0], data1.shape[1], data1.shape[2], 1 ) )
    data2 = np.reshape(data2, ( data2.shape[0], data2.shape[1], data2.shape[2], 1 ) )
    
    return( data1, data2, area_names[idx1], area_names[idx2], time_step, stim )



def get_subset2(sess_no, raw_path, 
               align_on1, from_time1, to_time1, window_size1, lowcut1, highcut1, 
               align_on2, from_time2, to_time2, window_size2, lowcut2, highcut2, 
               step, delay,
               epsillon = 100, order = 3,
               only_correct_trials = True, renorm = True ):

    tinfo_path = raw_path + 'trial_info.mat'
    rinfo_path = raw_path + 'recording_info.mat'
    
    # get all data
    data1_filtered, data2_filtered = get_preprocessed_from_raw2(sess_no, raw_path, 
                                              align_on1, from_time1 - epsillon, to_time1 + epsillon, lowcut1, highcut1, order, 
                                              align_on2, from_time2 - epsillon, to_time2 + epsillon, lowcut2, highcut2, order)


    
    # don't keep missing data // keep only_correct_trials if True
    
    responses = io.get_responses(tinfo_path)
    if only_correct_trials == False:
        ind_to_keep = (responses == responses).flatten()
    else:
        ind_to_keep = (responses == 1).flatten()
        
    #data1 =data1[ind_to_keep, :, :] # in the same time
    #data2 =data2[ind_to_keep, :, :]
    
    data1_filtered = data1_filtered[ind_to_keep,:,:]
    data2_filtered = data2_filtered[ind_to_keep,:,:]
    
    # select electrode
    area_names = io.get_area_names(rinfo_path)
    

            
    if epsillon != 0 :    
        data1_filtered = data1_filtered[:, :, epsillon : -epsillon ]
        data2_filtered = data2_filtered[:, :, epsillon : -epsillon ]
    else:
        data1_filtered = data1_filtered[:, :, :]
        data2_filtered = data2_filtered[:, :, :]
        

    i_max = min ( int(data1_filtered.shape[2]  - window_size1)//step, 
                int(data2_filtered.shape[2] - window_size2)//step ) +1
    ### Get list of step
    list_step = []
    for i in range(i_max):
        list_step = list_step + data1_filtered.shape[0] * [i *step]
    time_step = np.array(list_step)
    
    # get list of stim
    classes = 5
    targets = io.get_samples(tinfo_path)
    targets = targets[ind_to_keep].astype(int).flatten()
    targets =  np.tile(targets, i_max)
    stim = np.eye(classes)[targets].reshape(targets.shape[0], classes)
    
    # concat data with time step
    for i in range(i_max):
        if i ==0 :
            data1 = data1_filtered[:, 
                        :, 
                         + i*step : i*step + window_size1 ]
            
            data2 = data2_filtered[:, 
                        :, 
                        i*step : i*step + window_size2 ]
        else:
            data1 = np.concatenate( (data1, 
                                    data1_filtered[:, 
                                        :,
                                        i*step : i*step + window_size1 ]),
                                    axis =0 )
            data2 = np.concatenate( (data2, 
                                    data2_filtered[:, 
                                        :, 
                                        i*step : i*step + window_size2 ]) ,
                                    axis =0 )
                                    
    
    ### variable for shape
    #n_chans1 = len(elec1)
    #n_chans2 = len(elec2)
            
    #samples_per_trial1 = data1.shape[2] # = window_size1
    #samples_per_trial2 = data2.shape[2] # = window_size2
    
    # renorm data : mean = 0 and var = 1
    if renorm == True :
        data1 = pp.renorm(data1)
        data2 = pp.renorm(data2)

    ## change type and reshape
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)

    data1 = np.reshape(data1, ( data1.shape[0], data1.shape[1], data1.shape[2], 1 ) )
    data2 = np.reshape(data2, ( data2.shape[0], data2.shape[1], data2.shape[2], 1 ) )
    
    return( data1, data2, area_names, area_names, time_step, stim )



