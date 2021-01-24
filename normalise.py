import pandas as pd
import numpy as np







##-----------------------------------------------------------##

def normalise(lst):
    normalised_list = []
    max_val = np.max(lst)
    min_val = np.min(lst)
    
    for i in lst:
        norm_val = (i-min_val)/(max_val-min_val)
        normalised_list.append(norm_val)
    
    return normalised_list

##-----------------------------------------------------------##



## takes a numeric list and feature-scales. Must be numeric. ##
def make_normalised_list(df,concat):
        
    
    normalised_list = []
    max_val = np.max(concat)
    min_val = np.min(concat)
    
    for i in df:
        norm_val = (i-min_val)/(max_val-min_val)
        normalised_list.append(norm_val)
    
    return normalised_list
##-----------------------------------------------------------##

# input the df columns you want normalised. INput 
def hectic(*columns):
     
    
    concat_lst = []
    normalised_list = []
    
    for col_no in range(len(columns)):
        column = columns[col_no]
        column = column.tolist()
        
        for value in column:
            concat_lst.append(value)
   
    max_val = np.max(concat_lst)
    min_val = np.min(concat_lst)
    
    

           
    for i in concat_lst:
        
        norm_val = (i-min_val)/(max_val-min_val)
        normalised_list.append(norm_val)
   
    return normalised_list











## takes a normalised list and converts it back to original scaling ##
def unnormalise(df_norm, df_orig):
    
    unnormalised_list = []
    max_val = np.max(df_orig)
    min_val = np.min(df_orig)
    
    for i in df_norm:
        unnorm_val = (i*(max_val-min_val))+min_val
        unnormalised_list.append(unnorm_val)
        
    return unnormalised_list
##-------------------------------------------------------------------##


def sigmoid_list(x):
          
    lst = []

    for i in x:
        sig = np.exp(x)/(np.exp(x)+1)
        lst.append(sig)
        
    return lst






