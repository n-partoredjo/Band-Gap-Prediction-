

import numpy as np
import pandas as pd 
import random

file = pd.ExcelFile('making_zero_list.xlsx') 
df = file.parse('main')  

def delete_some_zeros(df,df_out,remove_perc):
      
    zeros_list = df_out ==0
    zero_indices = []   
    count = 0
    
    for i in zeros_list:
        if i == True :
            zero_indices.append(count)
        count += 1
    
    n = round(remove_perc*len(zero_indices))
    drop_indices = random.sample(zero_indices,n)   
    df_subset = df.drop(drop_indices)

    return df_subset


#a = delete_some_zeros(df,df['band_gap'],0.8)