import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size = 15)

from tolerance_characterisation_zero import discretize




data = pd.read_csv('df_for_data_distribution.csv')


BG_list = discretize(data['Band gap [eV]'], 1, False)



classes = np.unique(BG_list)

classes = np.concatenate([classes[-1:], classes[:-1]])
bins = len(classes)




dicts = {}


for c in classes:
    dicts[c] = 0


for BG in BG_list:    
    for c in dicts:              
        if BG == c:
            dicts[c] += 1
            
 
classes = np.char.replace(classes, "<= x <", ",").tolist()


cl = []

    
for i in classes[1:] :
    cl.append('{}{}{}'.format('[',i,')'))
    
#cl = np.concatenate([['zero'], cl])    

    
#this is for RF
cl = np.concatenate([cl,['zero']])    


# p = plt.figure()
# values = dicts.values()
# plt.bar(classes,values,color = 'g')
# plt.xticks(range(bins), cl, rotation = 45)
# plt.ylabel('No. Samples')
# plt.xlabel('Band Gap Range (eV)')
# plt.show()


