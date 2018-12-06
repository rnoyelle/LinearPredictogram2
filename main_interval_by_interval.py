import sys
import os
import os.path

import lib.cnn.matnpyio as io
#import lib.cnn.cnn as cnn 
import lib.cnn.matnpy as matnpy

import tensorflow as tf
import numpy as np
#from math import ceil

#import random
from sklearn.model_selection ifmport train_test_split
#from sklearn.model_selection import StratifiedKFold

import pandas as pd
import datetime


import lib.cnn.preprocess as pp


#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import os


################################################
#################### PARAMS ####################
################################################

#################
### base path ###
#################

# path to raw_path :  base_path/sess_no/session01/file.mat
base_path = '/media/rudy/disk2/lucy/' 

# path where to save the result
result_path = '/home/rudy/Python2/regression_linear2/result_ridge/'


session = sorted(os.listdir(base_path))
session.remove('unique_recordings.mat')
print(session)




#### 
dico_aro_area_to_cortex = io.get_dico_area_to_cortex()

for sess_no in session :  
    print(sess_no)
    
    ##############
    ###  PATH  ###
    ##############
    
    raw_path = base_path +sess_no+'/session01/'
    rinfo_path = base_path +sess_no+'/session01/' + 'recording_info.mat'
    
    ################################
    ###  PRE-PROCESSING PARAMS   ###
    ################################

    only_correct_trials = True

    align_on, from_time, to_time = 'sample', -700, 500 + 1000 
    lowcut, highcut, order = 7, 12, 3
    
    window_size = 200
    step = 100
    renorm = True
    
    ####################
    ### SPLIT PARAMS ###
    ####################
    
    seed = np.random.randint(1,10000)
    
    train_size = 0.8
    test_size = 1 - train_size
    
    ####################
    ### RIDGE PARAMS ###
    ####################
    
    alpha = 0.1 # >0.1 is good
    
    

    
    ##################
    ### LOAD DATA  ###
    ##################
    
    
    # get list of area 
    target_areas = []
    for cortex in ['Visual', 'Prefrontal', 'Motor', 'Parietal', 'Somatosensory'] : 
        areas = io.get_area_cortex(rinfo_path, cortex, unique = True)
        for area in areas :
            target_areas.append(area)
            
            
    data, area_names = matnpy.get_subset_by_areas(sess_no, raw_path, 
                        align_on, from_time, to_time, lowcut, highcut, target_areas,
                        epsillon=100, order=order,
                        only_correct_trials = only_correct_trials, renorm=renorm)
    
    print(data.shape)
    
    
    
    
    for area1 in target_areas :
        # get idx of area1
        idx1 = [ count for count, area in enumerate(area_names) if area == area1]
        
        for count1, idx_channel1 in enumerate(idx1):
            
            
            for area2 in target_areas :
                # get idx of area2
                idx2 =  [ count for count, area in enumerate(area_names) if area == area2]
                
                for count2, idx_channel2 in enumerate(idx2):
                    
                    print(area1, count1, area2, count2)
                    
                    for n_step in range( int( ( to_time - from_time - window_size)/step) +1 ) :
                        
                        print('align_on', align_on, 'from', n_step * step +from_time , 'to', n_step*step + window_size + from_time)
                        
                        
                        ##################
                        ### GET SUBSET ###
                        ##################
                        
                        X = data[:,[idx_channel1], n_step * step : n_step*step + window_size] # shape = (trial, n_chans, n_time ) 
                        Y = data[:,[idx_channel2], n_step * step : n_step*step + window_size] # shape = (trial, n_chans, n_time )
                        
                        n_chans1 = X.shape[1] # = 1 in that case
                        n_chans2 = Y.shape[1] # = 1 in that case
                        
                        
                        if renorm == True :
                            X = pp.renorm(X)
                            Y = pp.renorm(Y)
                            
                        X = np.reshape(X, (X.shape[0], -1)) # shape = (trial, features)
                        Y = np.reshape(Y, (Y.shape[0], -1)) # shape = (trial, features)
                        
                        ################################################
                        #         TRAINING AND TEST REGRESSION         #
                        ################################################
                        
                        #####################
                        ### SPLIT DATASET ###
                        #####################
                        
                        indices = [i for i in range(data.shape[0])]
                        
                        x_train, x_test, y_train, y_test, ind_train, ind_test = (
                            train_test_split(
                                X, 
                                Y, 
                                indices,
                                test_size=test_size, 
                                random_state=seed
                                )
                            ) 
                                       
                        ###############################
                        ### TRAIN linear regression ###
                        ###############################
                        
                        #reg = LinearRegression().fit(x_train, y_train)
                        reg = Ridge(alpha=alpha).fit(x_train, y_train)
                        
                        
                        ##############################
                        ### TEST linear regression ###
                        ##############################
                                                
                        # on trainning base
                        r2_train = reg.score(x_train, y_train)
                        # on testing base
                        r2_test = reg.score(x_test, y_test)
                        
                        # error bar
                        
                        y_train_predict = reg.predict(x_train)
                        mse = np.mean( (y_train - y_train_predict)**2, axis = 1 )
                        r2_train_error_bar = np.std(mse)/(np.var(y_train) * np.sqrt( mse.shape[0]) )
                        
                        y_test_predict = reg.predict(x_test)
                        mse = np.mean( (y_test - y_test_predict)**2, axis = 1 )
                        r2_test_error_bar = np.std(mse)/(np.var(y_test) * np.sqrt( mse.shape[0]) )
                        
                        ###################
                        ### SAVE RESUTS ###
                        ###################
                        
                        cortex1 = dico_aro_area_to_cortex[area1]
                        cortex2 = dico_aro_area_to_cortex[area2]
                        str_freq = 'low'+str(lowcut)+'high'+str(highcut)+'order'+str(order)
                        interval = 'align_on_' + align_on + '_from_time_' + str(from_time +  n_step * step) +'_to_time_'+ str(from_time +  n_step * step + window_size)

                        data_tuning = [ sess_no, area1, count1, area2, count2,   
                                        cortex1, cortex2,  
                                        str_freq,
                                        interval,
                                        window_size, 
                                        len(ind_test), len(ind_train), 
                                        alpha,
                                        n_chans1, n_chans2, 
                                        only_correct_trials, 
                                        r2_train, r2_test,
                                        r2_train_error_bar, r2_test_error_bar,
                                        renorm, seed] 


                        df = pd.DataFrame([data_tuning],
                                        columns=[ 'session', 'area1','num1', 'area2', 'num2',   
                                        'cortex1', 'cortex2',  
                                        'str_freq',  
                                        'interval',
                                        'window_size',   
                                        'len(ind_test)', 'len(ind_train)', 
                                        'alpha',
                                        'n_chans1', 'n_chans2', 
                                        'only_correct_trials', 
                                        'r2_train', 'r2_test', 
                                        'r2_train_error_bar', 'r2_test_error_bar',
                                        'renorm', 'seed'] ,
                                        index=[0])


                        file_name = result_path + 'result_sess_no_'+str(sess_no)+'channel_to_channel_all_interval_ridge.csv'
                        file_exists = os.path.isfile(file_name)
                        if file_exists :
                            with open(file_name, 'a') as f:
                                df.to_csv(f, mode ='a', index=False, header=False)
                        else:
                            with open(file_name, 'w') as f:
                                df.to_csv(f, mode ='w', index=False, header=True)
 
                                

                                

 
