import sys
import os
import os.path

import lib.cnn.matnpyio as io
#import lib.cnn.cnn as cnn 
import lib.cnn.matnpy as matnpy

import tensorflow as tf
import numpy as np
from math import ceil

import random
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold

import pandas as pd
import datetime


import lib.cnn.preprocess as pp


from sklearn.linear_model import LinearRegression

import os


################################################
#################### PARAMS ####################
################################################

#################
### base path ###
#################

base_path = '/media/rudy/disk2/lucy/'

###################
### data params ###
###################

session = os.listdir(base_path)
session.remove('unique_recordings.mat')
print(session)


#### 
dico_aro_area_to_cortex = io.get_dico_area_to_cortex()

for sess_no in session :  
    print(sess_no)
    

    # path
    raw_path = base_path +sess_no+'/session01/'
    rinfo_path = base_path +sess_no+'/session01/' + 'recording_info.mat'

    only_correct_trials = True

    align_on, from_time, to_time = 'sample', -700, 500 + 1000 
    lowcut, highcut, order = 7, 12, 3
    
    window_size = 200
    step = 100
    
    
    # get list of area 
    target_areas = []
    for cortex in ['Visual', 'Prefrontal', 'Motor', 'Parietal', 'Somatosensory'] : 
        areas = io.get_area_cortex(rinfo_path, cortex1, unique = True)
        for area in areas :
            target_areas.append(area)
            
            
    data, area_names = matnpy.get_subset_by_areas(sess_no, raw_path, 
                        align_on, from_time, to_time, lowcut, highcut, target_areas,
                        epsillon=100, order=order,
                        only_correct_trials = only_correct_trials, renorm=renorm)
    
    
    for area1 in target_areas :
        # get idx of area1
        idx1 = [ count for count, area in enumerate(area_names) if area == area1]
        
        for count1, idx_channel1 in enumerate(idx1):
            
            X = data[:,idx_channel1, n_step * step : n_step*step + window_size]
            
            for area2 in target_areas :
                # get idx of area2
                idx2 =  [ count for count, area in enumerate(area_names) if area == area2]
                
                for count2, idx_channel2 in enumerate(idx2):
                    
                    for n_step in range( int( ( to_time - from_time - window_size)/step) +1 ) :
                
                        X = data[:,[idx_channel1], n_step * step : n_step*step + window_size] # shape = (trial, n_chans, n_time ) 
                        Y = data[:,[idx_channel2], n_step * step : n_step*step + window_size] # shape = (trial, n_chans, n_time )
                        
                        n_chans1 = X.shape[1]
                        n_chans2 = Y.shape[1]
                        
                        if renorm == True :
                            X = pp.renorm(X)
                            Y = pp.renorm(Y)
                            
                        X = np.reshape(X, (X.shape[0], -1))
                        Y = np.reshape(Y, (Y.shape[0], -1))
                        
                        ################################################
                        #         TRAINING AND TEST NETWORK            #
                        ################################################
                        
                        ### SPLIT
                        indices = [i for i in range(data1.shape[0])]
                        
                        x_train, x_test, y_train, y_test, ind_train, ind_test = (
                            train_test_split(
                                X, 
                                Y, 
                                indices,
                                test_size=test_size, 
                                random_state=seed
                                )
                            ) 
                                       
                        ### TRAIN linear regression
                        reg = LinearRegression().fit(x_train, y_train)
                        
                        # test linear regression
                        
                        # on trainning base
                        r2_train = reg.score(x_train, y_train)
                        # on testing base
                        r2_test = reg.score(x_test, y_test)
                        
                        # error bar
                        
                        y_train_predict = reg.predict(x_train)
                        mse = np.mean( (y_train - y_train_predict)**2, axis = 1 )
                        r2_train_error_bar = np.std(mse)/np.var(y_train)
                        
                        y_test_predict = reg.predict(x_test)
                        mse = np.mean( (y_test - y_test_predict)**2, axis = 1 )
                        r2_test_error_bar = np.std(mse)/np.var(y_test)
                        
                        ### SAVE RESUTS
                        
                        cortex1 = dico_aro_area_to_cortex[area1]
                        cortex2 = dico_aro_area_to_cortex[area2]
                        str_freq = 'low'+str(lowcut)+'high'+str(highcut)+'order'+str(order)
                        interval = 'align_on_' + align_on + 'from_time_' + str(from_time +  n_step * step) +'_to_time_'+ str(from_time +  n_step * step + window_size)

                        data_tuning = [ sess_no, area1, count1, area2, count2,   
                                        cortex1, cortex2,  
                                        str_freq,
                                        window_size, 
                                        len(ind_test), len(ind_train), 
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
                                        'n_chans1', 'n_chans2', 
                                        'only_correct_trials', 
                                        'r2_train', 'r2_test', 
                                        'r2_train_error_bar', 'r2_test_error_bar',
                                        'renorm', 'seed'] ,
                                        index=[0])


                        file_name = '/home/rudy/Python2/regression_linear2/result/' + 'result_sess_no_'+str(sess_no)+'channel_to_channel_all_interval_with_error_bar.csv'
                        file_exists = os.path.isfile(file_name)
                        if file_exists :
                            with open(file_name, 'a') as f:
                                df.to_csv(f, mode ='a', index=False, header=False)
                        else:
                            with open(file_name, 'w') as f:
                                df.to_csv(f, mode ='w', index=False, header=True)
                                
                                
                                
for sess_no in session :  
    print(sess_no)
    

    # path
    raw_path = base_path +sess_no+'/session01/'
    rinfo_path = base_path +sess_no+'/session01/' + 'recording_info.mat'

    only_correct_trials = True

    align_on, from_time, to_time = 'match', -500, 0 + 1900 
    lowcut, highcut, order = 7, 12, 3
    
    window_size = 200
    step = 100
    
    
    # get list of area 
    target_areas = []
    for cortex in ['Visual', 'Prefrontal', 'Motor', 'Parietal', 'Somatosensory'] : 
        areas = io.get_area_cortex(rinfo_path, cortex1, unique = True)
        for area in areas :
            target_areas.append(area)
            
            
    data, area_names = matnpy.get_subset_by_areas(sess_no, raw_path, 
                        align_on, from_time, to_time, lowcut, highcut, target_areas,
                        epsillon=100, order=order,
                        only_correct_trials = only_correct_trials, renorm=renorm)
    
    
    for area1 in target_areas :
        # get idx of area1
        idx1 = [ count for count, area in enumerate(area_names) if area == area1]
        
        for count1, idx_channel1 in enumerate(idx1):
            
            X = data[:,idx_channel1, n_step * step : n_step*step + window_size]
            
            for area2 in target_areas :
                # get idx of area2
                idx2 =  [ count for count, area in enumerate(area_names) if area == area2]
                
                for count2, idx_channel2 in enumerate(idx2):
                    
                    for n_step in range( int( ( to_time - from_time - window_size)/step) +1 ) :
                
                        X = data[:,[idx_channel1], n_step * step : n_step*step + window_size] # shape = (trial, n_chans, n_time ) 
                        Y = data[:,[idx_channel2], n_step * step : n_step*step + window_size] # shape = (trial, n_chans, n_time )
                        
                        n_chans1 = X.shape[1]
                        n_chans2 = Y.shape[1]
                        
                        if renorm == True :
                            X = pp.renorm(X)
                            Y = pp.renorm(Y)
                            
                        X = np.reshape(X, (X.shape[0], -1))
                        Y = np.reshape(Y, (Y.shape[0], -1))
                        
                        ################################################
                        #         TRAINING AND TEST NETWORK            #
                        ################################################
                        
                        ### SPLIT
                        indices = [i for i in range(data1.shape[0])]
                        
                        x_train, x_test, y_train, y_test, ind_train, ind_test = (
                            train_test_split(
                                X, 
                                Y, 
                                indices,
                                test_size=test_size, 
                                random_state=seed
                                )
                            ) 
                                       
                        ### TRAIN linear regression
                        reg = LinearRegression().fit(x_train, y_train)
                        
                        # test linear regression
                        
                        # on trainning base
                        r2_train = reg.score(x_train, y_train)
                        # on testing base
                        r2_test = reg.score(x_test, y_test)
                        
                        # error bar
                        
                        y_train_predict = reg.predict(x_train)
                        mse = np.mean( (y_train - y_train_predict)**2, axis = 1 )
                        r2_train_error_bar = np.std(mse)/np.var(y_train)
                        
                        y_test_predict = reg.predict(x_test)
                        mse = np.mean( (y_test - y_test_predict)**2, axis = 1 )
                        r2_test_error_bar = np.std(mse)/np.var(y_test)
                        
                        ### SAVE RESUTS
                        
                        cortex1 = dico_aro_area_to_cortex[area1]
                        cortex2 = dico_aro_area_to_cortex[area2]
                        str_freq = 'low'+str(lowcut)+'high'+str(highcut)+'order'+str(order)
                        interval = 'align_on_' + align_on + 'from_time_' + str(from_time +  n_step * step) +'_to_time_'+ str(from_time +  n_step * step + window_size)

                        data_tuning = [ sess_no, area1, count1, area2, count2,   
                                        cortex1, cortex2,  
                                        str_freq,
                                        window_size, 
                                        len(ind_test), len(ind_train), 
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
                                        'n_chans1', 'n_chans2', 
                                        'only_correct_trials', 
                                        'r2_train', 'r2_test', 
                                        'r2_train_error_bar', 'r2_test_error_bar',
                                        'renorm', 'seed'] ,
                                        index=[0])


                        file_name = '/home/rudy/Python2/regression_linear2/result/' + 'result_sess_no_'+str(sess_no)+'channel_to_channel_all_interval_with_error_bar.csv'
                        file_exists = os.path.isfile(file_name)
                        if file_exists :
                            with open(file_name, 'a') as f:
                                df.to_csv(f, mode ='a', index=False, header=False)
                        else:
                            with open(file_name, 'w') as f:
                                df.to_csv(f, mode ='w', index=False, header=True)
                                
                                
                                

 
