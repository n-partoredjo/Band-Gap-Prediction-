def make_corr_list(corr_mat, params, threshold):
       
    col_index = 0
    corr_lst = []   
    lst1 = []
    lst2 = []
        
    for parameter_name in params:       
        row_index = 0 
        
        # assigns variable to list of all column entries
        full_col = corr_mat[parameter_name].values.tolist()
        
        # cuts full_col down to below the leading diagonal in the matrix 
        reduced_col = full_col[col_index+1:]
        
        col_index += 1
        
        for i in range(len(reduced_col)):            
            row_index = i + col_index+1           
            value = reduced_col[i]
            
            if value > threshold:
                
                lst1.append(parameter_name)
                lst2.append(params[row_index-1])                                
                line = '%s correlates with %s'
                features = (parameter_name,) + (params[row_index-1],)
                lst_entry = line %features                
                corr_lst.append(lst_entry)
        
            row_index += 1 
            
    return corr_lst


def make_neg_corr_list(corr_mat, params, threshold):
    
    col_index = 0
    corr_lst = []  
    lst1 = []
    lst2 = []
      
    for parameter_name in params:
        
        row_index = 0 
        
        # assigns variable to list of all column entries
        full_col = corr_mat[parameter_name].values.tolist()
        
        # cuts full_col down to below the leading diagonal in the matrix 
        reduced_col = full_col[col_index+1:]
        
        col_index += 1
        
        for i in range(len(reduced_col)):            
            row_index = i + col_index+1
            value = reduced_col[i]
            
            if value < threshold:
                
                lst1.append(parameter_name)
                lst2.append(params[row_index-1])                
                line = '%s negatively correlates with %s'
                features = (parameter_name,) + (params[row_index-1],)
                lst_entry = line %features                
                corr_lst.append(lst_entry)
        
            row_index += 1 
            
    return corr_lst