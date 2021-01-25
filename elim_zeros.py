


import pandas as pd 
import random
import numpy as np

file1 = pd.ExcelFile('elim_zeros_test.xlsx')
df = file1.parse('main') 



def delete_label_zeros(df,df_out,remove_perc):
     
        
    zeros_list = df_out == 'zero'
    zeros_ind = zeros_list.index[zeros_list].tolist()
    
#    print(zeros_ind)
    
    
    n = round(remove_perc*len(zeros_ind))
    #print(n)
    
    drop_indices = random.sample(zeros_ind,n)   
    df_subset = df.drop(drop_indices)
    
    
    return df_subset




def delete_number_zeros(test_X,test_Y, remove_perc):
    
    
    
    
    zeros_list = test_Y==0
    
#    print(zeros_list)
    zeros_ind = zeros_list.index[zeros_list].tolist()  
#    print(zero_indices)
    
    
    n = round(remove_perc*len(zeros_ind))
    #print(n)
    
    drop_indices = random.sample(zeros_ind,n) 
    test_X_subset = test_X.drop(drop_indices)
    test_Y_subset = test_Y.drop(drop_indices)
    

    return test_X_subset,test_Y_subset
#
#
#
#
#
#t = [0,0,0,0,0,0,1,1,1,0,0]
#
#test = delete_zeros_from_list(t, 0.5)


#a = delete_label_zeros(df,df['label'],0.5)
#
#
#print(a)
#print(a['label'])

'''

df = test
df_out = test['label']

'''