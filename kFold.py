import pandas as pd
import numpy as np
import os
import sys
import csv
import plotly.express as px

# Suppress warnings that are a result of current diasgreements between pandas/numpy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
Below are two functions used in the creation of kfold validation sets.
'''
def getStratifiedRandIndexArrSplit(cur_df, num_folds, class_col):
    '''
    Inputs:
    cur_df - dataset to be randomized
    num_folds - number of folds for k-fold cross validation
    class_col - column label that contains class
    '''
    train_and_test_index_arrays = {}
    for ind in range(0, num_folds):
        train_and_test_index_arrays[f'arr_{ind}'] = []
    # Separate out entries by class, shuffle those df
    class_list = cur_df[class_col].unique()
    for val in class_list:
        class_df = cur_df[cur_df[class_col]==val]
        class_df = class_df.iloc[np.random.permutation(len(class_df))]
        shuffled_index = list(class_df.index.unique())
        len_df = len(class_df)
        
        arr_ind=0
        while len(shuffled_index)>0:
            ind_val = shuffled_index.pop(0)
            train_and_test_index_arrays[f'arr_{arr_ind}'].append(ind_val)
            if arr_ind == num_folds-1:
                arr_ind = 0
            else:
                arr_ind+=1        

    for arr_key in train_and_test_index_arrays:
        train_and_test_index_arrays[arr_key].sort()
    
    return train_and_test_index_arrays

def getRandomizedIndexArrSplit(cur_df, num_folds):
    # Begin by randomizing the entries. 
    cur_df_shuffled = cur_df.iloc[np.random.permutation(len(cur_df))]
    shuffled_index = list(cur_df_shuffled.index.unique())
    
    # Find the number of entries for each of the folds, based on num_folds
    len_df = len(cur_df)
    num_items_per_fold = int(len_df/num_folds)
    
    train_and_test_index_arrays = {}
    for ind in range(0,num_folds):
        train_and_test_index_arrays[f'arr_{ind}'] = []
    
    arr_ind = 0
    while len(shuffled_index)>0:
        ind_val = shuffled_index.pop(0)
        train_and_test_index_arrays[f'arr_{arr_ind}'].append(ind_val)
        if arr_ind == num_folds-1:
            arr_ind = 0
        else:
            arr_ind+=1
    
    for arr_key in train_and_test_index_arrays:
        train_and_test_index_arrays[arr_key].sort()
        

    return train_and_test_index_arrays

def getKFoldValidationSets(cur_df, randomized_arrays, create_validation_set=False):
    kfold_sets={}
    kfold_iter = 0
    
    for ind in range(0, len(randomized_arrays)):
        # Create a temp_ds with a placeholder
        temp_ds = cur_df.assign(kfold_label="")



        # Set the first randomized array as the test set
        train_const=['test' for val in range(0,len(randomized_arrays[f'arr_{kfold_iter}']))]
        train_series = pd.Series(data=train_const,index=randomized_arrays[f'arr_{kfold_iter}'], name='kfold_label')

        if create_validation_set:
            # Check if we're on the last array: if so, use arr_0 as validation array
            if kfold_iter==len(randomized_arrays)-1:
                val_iter = 0
            else:
                val_iter = kfold_iter+1

            val_const=['validate' for val in range(0,len(randomized_arrays[f'arr_{val_iter}']))]
            validate_series=pd.Series(data=val_const,index=randomized_arrays[f'arr_{val_iter}'], name='kfold_label')

            full_series = train_series.append(validate_series)

            for ind_2 in range(0, len(randomized_arrays)):
                if ind_2!=kfold_iter and ind_2!=val_iter:
                    test_const=['train' for val in range(0,len(randomized_arrays[f'arr_{ind_2}']))]
                    test_series=pd.Series(data=test_const,index=randomized_arrays[f'arr_{ind_2}'], name='kfold_label')

                    full_series = full_series.append(test_series)
    
        else:
            full_series = train_series

            for ind_2 in range(0, len(randomized_arrays)):
                if ind_2!=kfold_iter:
                    test_const=['train' for val in range(0,len(randomized_arrays[f'arr_{ind_2}']))]
                    test_series=pd.Series(data=test_const,index=randomized_arrays[f'arr_{ind_2}'], name='kfold_label')

                    full_series = full_series.append(test_series)
        kfold_iter+=1
        
        full_series = full_series.sort_index()
        
        temp_ds['kfold_label'] = full_series 
        
        kfold_sets[f'kfold_{ind}'] = temp_ds    
        
    return kfold_sets
    
