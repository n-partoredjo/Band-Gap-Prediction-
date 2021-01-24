import pandas as pd 
import numpy as np

### takes a list of discrete labelled data and makes into a  numeric code ###

def make_numeric(df):
    
    numeric_list = pd.factorize(df, sort = True)[0] 
     
    return numeric_list


### makes a table showing the discrete data and it's numerical code ###

def make_ref_table(df):
    
    labels = np.unique(df)
    code = np.unique(pd.factorize(df, sort = True)[0])
    ref_table = pd.DataFrame(list(zip(labels, code)), 
               columns =['label', 'code'])
    
    return ref_table

### makes a table comparing the actual and predicted values ###

def make_comp_table(actual, predicted):
    
    list1 = actual
    list2 = predicted
               
    comp_table = pd.DataFrame(list(zip(list1, list2)), 
               columns =['Actual', 'Predicted'])
    
    comp_table['Correct?'] = comp_table['Actual'] == comp_table['Predicted']
    accuracy = comp_table['Correct?'].mean()
    
    return comp_table


### finds the accuracy of the classifier ###

def find_accuracy(actual, predicted):
    
    list1 = actual
    list2 = predicted
               
    comp_table = pd.DataFrame(list(zip(list1, list2)), 
               columns =['Actual', 'Predicted'])
    
    comp_table['Correct?'] = comp_table['Actual'] == comp_table['Predicted']
    accuracy = comp_table['Correct?'].mean()
    
    return accuracy

test = make_comp_table([1,2,3], [1,4,3])
accuracy_test = find_accuracy([1,2,3], [1,4,3])

