
def how_many_above(cutoff,lst):
    
    how_many  = sum(i > cutoff for i in lst)
    
    return how_many



perf = [1,2,3,4,5,6,7,9]

test = how_many_above(5, perf)
