import pandas as pd 

file = pd.ExcelFile('tolerance_test.xlsx') 
df = file.parse('main') 


values = df['values'].values.tolist()

tol = 0.5


# converts a list of numbers into a list of discrete ranges
# if zero_split is True, it just generates a list of 'zero' and 'non-zero'

def discretize(values, tol, zero_split):
    
    counter = 0
    discrete_list = []

    for i in values:
        
        if i < 0.0001:
            discrete_list.append('zero')
        elif zero_split == True:
            
            discrete_list.append('non-zero')
            
        else:  
            while i >= counter:
                
                counter +=tol
                             
            discrete_list.append("{} <= x < {}".format((round(counter,2) - tol),
                                 round(counter,2)))
            
            counter = 0   

    return discrete_list 





def discretize_num(values, zero_split, tol = 0.1):
    
    counter = 0
    discrete_list = []

    for i in values:
        
        if i < 0.0001:
            discrete_list.append(0)
        elif zero_split == True:
            
            discrete_list.append(1)
            
        else:  
            while i >= counter:
                
                counter +=tol
                             
            discrete_list.append("{} <= x < {}".format((round(counter,2) - tol),
                                 round(counter,2)))
            
            counter = 0   

    return discrete_list 





ranges = discretize(values, tol,True)



            
    

