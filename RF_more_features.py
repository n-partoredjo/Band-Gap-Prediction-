import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size = 15)


## install sklearn packages ##
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

## import helper functions ##
from correlation_list import make_corr_list, make_neg_corr_list
from normalise import make_normalised_list, unnormalise
from delete_some_zeros import delete_some_zeros
from make_numeric_codes import make_numeric, make_ref_table, find_accuracy, make_comp_table
from tolerance_characterisation_zero import discretize
from masked_list import make_masked_list
from elim_zeros_3 import delete_label_zeros
from how_many_above import how_many_above

## read excel files ##
file1 = pd.ExcelFile('good_data_set.xlsx')
df = file1.parse('main') 





## ------- Pre-processing Begins --------------- ##


# delete a certain percentage of the zero band gaps #
df = delete_some_zeros(df,df['Band gap [eV]'], remove_perc = 0)


# Band Gap #

bg_orig = df['Band gap [eV]']




# Formation Energy #
FE_orig = df['Formation energy [eV/atom]']
df['Formation energy [eV/atom]'] = make_normalised_list(df['Formation energy [eV/atom]'],df['Formation energy [eV/atom]'])


## create normalised columns and remove older column ##
## _comb and _concat combine columns to normalise over the full range of values ##


# alpha, gamma and beta [deg] parameters #
angle_comb = (df['gamma [deg]'],df['beta [deg]'],df['alpha [deg]'])
angle_concat = pd.concat(angle_comb)
df['alpha [deg]'] = make_normalised_list(df['alpha [deg]'],angle_concat)
df['beta [deg]'] = make_normalised_list(df['beta [deg]'],angle_concat)
df['gamma [deg]'] = make_normalised_list(df['gamma [deg]'],angle_concat)



# a, b and c [ang] #
abc_comb = (df['a [ang]'],df['b [ang]'],df['c [ang]'])
abc_concat = pd.concat(abc_comb)
df['a [ang]'] = make_normalised_list(df['a [ang]'],abc_concat)
df['b [ang]'] = make_normalised_list(df['b [ang]'],abc_concat)
df['c [ang]'] = make_normalised_list(df['c [ang]'],abc_concat)


# Radius of A and B [ang] #
rad_comb = (df['Radius A [ang]'],df['Radius B [ang]'])
rad_concat = pd.concat(rad_comb)
df['Radius A [ang]'] = make_normalised_list(df['Radius A [ang]'],rad_concat)
df['Radius B [ang]'] = make_normalised_list(df['Radius B [ang]'],rad_concat)



# Valence of A and B #
valence_comb = (df['Valence A'],df['Valence B'])
valence_concat = pd.concat(valence_comb)
df['Valence A'] = make_normalised_list(df['Valence A'],valence_concat)
df['Valence B'] = make_normalised_list(df['Valence B'],valence_concat)


# Atomic number #
atomnum_comb = (df['A_atomic_number'],df['B_atomic_number'])
atomnum_concat = pd.concat(atomnum_comb)
df['A_atomic_number'] = make_normalised_list(df['A_atomic_number'],atomnum_concat)
df['B_atomic_number'] = make_normalised_list(df['B_atomic_number'],atomnum_concat)



# Atomic Radius #
atomrad_comb = (df['A_atomic_radius'],df['B_atomic_radius'])
atomrad_concat = pd.concat(atomrad_comb)
df['A_atomic_radius'] = make_normalised_list(df['A_atomic_radius'],atomrad_concat)
df['B_atomic_radius'] = make_normalised_list(df['B_atomic_radius'],atomrad_concat)



# Atomic Volume #
atomvol_comb = (df['A_atomic_volume'],df['B_atomic_volume'])
atomvol_concat = pd.concat(atomvol_comb)
df['A_atomic_volume'] = make_normalised_list(df['A_atomic_volume'],atomvol_concat)
df['B_atomic_volume'] = make_normalised_list(df['B_atomic_volume'],atomvol_concat)


# Boiling Point #
bp_comb = (df['A_boiling_point'],df['B_boiling_point'])
bp_concat = pd.concat(bp_comb)
df['A_boiling_point'] = make_normalised_list(df['A_boiling_point'],bp_concat)
df['B_boiling_point'] = make_normalised_list(df['B_boiling_point'],bp_concat)


# Density #
dens_comb = (df['A_density'],df['B_density'])
dens_concat = pd.concat(dens_comb)
df['A_density'] = make_normalised_list(df['A_density'],dens_concat)
df['B_density'] = make_normalised_list(df['B_density'],dens_concat)

# Dipole Polarizability #
dip_comb = (df['A_dipole_polarizability'],df['B_dipole_polarizability'])
dip_concat = pd.concat(dip_comb)
df['A_dipole_polarizability'] = make_normalised_list(df['A_dipole_polarizability'],dip_concat)
df['B_dipole_polarizability'] = make_normalised_list(df['B_dipole_polarizability'],dip_concat)


# Lattice Constant #
latt_comb = (df['A_lattice_constant'],df['B_lattice_constant'])
latt_concat = pd.concat(latt_comb)
df['A_lattice_constant'] = make_normalised_list(df['A_lattice_constant'],latt_concat)
df['B_lattice_constant'] = make_normalised_list(df['B_lattice_constant'],latt_concat)


# Melting Point #
melt_comb = (df['A_melting_point'],df['B_melting_point'])
melt_concat = pd.concat(melt_comb)
df['A_melting_point'] = make_normalised_list(df['A_melting_point'],melt_concat)
df['B_melting_point'] = make_normalised_list(df['B_melting_point'],melt_concat)

# Specific Heat #
spec_comb = (df['A_specific_heat'],df['B_specific_heat'])
spec_concat = pd.concat(spec_comb)
df['A_specific_heat'] = make_normalised_list(df['A_specific_heat'],spec_concat)
df['B_specific_heat'] = make_normalised_list(df['B_specific_heat'],spec_concat)


# vdw Radius #
vdw_comb = (df['A_vdw_radius'],df['B_vdw_radius'])
vdw_concat = pd.concat(vdw_comb)
df['A_vdw_radius'] = make_normalised_list(df['A_vdw_radius'],vdw_concat)
df['B_vdw_radius'] = make_normalised_list(df['B_vdw_radius'],vdw_concat)


# Covalent Radius Cordero #
cord_comb = (df['A_covalent_radius_cordero'],df['B_covalent_radius_cordero'])
cord_concat = pd.concat(cord_comb)
df['A_covalent_radius_cordero'] = make_normalised_list(df['A_covalent_radius_cordero'],cord_concat)
df['B_covalent_radius_cordero'] = make_normalised_list(df['B_covalent_radius_cordero'],cord_concat)

# Covalent Radius Pyykko #
pyyk_comb = (df['A_covalent_radius_pyykko'],df['B_covalent_radius_pyykko'])
pyyk_concat = pd.concat(pyyk_comb)
df['A_covalent_radius_pyykko'] = make_normalised_list(df['A_covalent_radius_pyykko'],pyyk_concat)
df['B_covalent_radius_pyykko'] = make_normalised_list(df['B_covalent_radius_pyykko'],pyyk_concat)

# Covalent Radius Pyykko Double #
pyyk2_comb = (df['A_covalent_radius_pyykko_double'],df['B_covalent_radius_pyykko_double'])
pyyk2_concat = pd.concat(pyyk2_comb)
df['A_covalent_radius_pyykko_double'] = make_normalised_list(df['A_covalent_radius_pyykko_double'],pyyk2_concat)
df['B_covalent_radius_pyykko_double'] = make_normalised_list(df['B_covalent_radius_pyykko_double'],pyyk2_concat)

# Covalent Radius Slater #
slat_comb = (df['A_covalent_radius_slater'],df['B_covalent_radius_slater'])
slat_concat = pd.concat(slat_comb)
df['A_covalent_radius_slater'] = make_normalised_list(df['A_covalent_radius_slater'],slat_concat)
df['B_covalent_radius_slater'] = make_normalised_list(df['B_covalent_radius_slater'],slat_concat)

# Pauling Electronegativity #
paul_comb = (df['A_en_pauling'],df['B_en_pauling'])
paul_concat = pd.concat(paul_comb)
df['A_en_pauling'] = make_normalised_list(df['A_en_pauling'],paul_concat)
df['B_en_pauling'] = make_normalised_list(df['B_en_pauling'],paul_concat)


# Heat of Formation #
heat_comb = (df['A_heat_of_formation'],df['B_heat_of_formation'])
heat_concat = pd.concat(heat_comb)
df['A_heat_of_formation'] = make_normalised_list(df['A_heat_of_formation'],heat_concat)
df['B_heat_of_formation'] = make_normalised_list(df['B_heat_of_formation'],heat_concat)

# vdw Radius uff #
uff_comb = (df['A_vdw_radius_uff'],df['B_vdw_radius_uff'])
uff_concat = pd.concat(uff_comb)
df['A_vdw_radius_uff'] = make_normalised_list(df['A_vdw_radius_uff'],uff_concat)
df['B_vdw_radius_uff'] = make_normalised_list(df['B_vdw_radius_uff'],uff_concat)

# vdw Radius Alvarez #
alva_comb = (df['A_vdw_radius_alvarez'],df['B_vdw_radius_alvarez'])
alva_concat = pd.concat(alva_comb)
df['A_vdw_radius_alvarez'] = make_normalised_list(df['A_vdw_radius_alvarez'],alva_concat)
df['B_vdw_radius_alvarez'] = make_normalised_list(df['B_vdw_radius_alvarez'],alva_concat)

# vdw Radius mm3 #
mm3_comb = (df['A_vdw_radius_mm3'],df['B_vdw_radius_mm3'])
mm3_concat = pd.concat(uff_comb)
df['A_vdw_radius_mm3'] = make_normalised_list(df['A_vdw_radius_mm3'],mm3_concat)
df['B_vdw_radius_mm3'] = make_normalised_list(df['B_vdw_radius_mm3'],mm3_concat)


# Ghosh Electronegativity #
ghosh_comb = (df['A_en_ghosh'],df['B_en_ghosh'])
ghosh_concat = pd.concat(ghosh_comb)
df['A_en_ghosh'] = make_normalised_list(df['A_en_ghosh'],ghosh_concat)
df['B_en_ghosh'] = make_normalised_list(df['B_en_ghosh'],ghosh_concat)

# c6 gb #
c6gb_comb = (df['A_c6_gb'],df['B_c6_gb'])
c6gb_concat = pd.concat(c6gb_comb)
df['A_c6_gb'] = make_normalised_list(df['A_c6_gb'],c6gb_concat)
df['B_c6_gb'] = make_normalised_list(df['B_c6_gb'],c6gb_concat)


# Atomic Weight #
weigh_comb = (df['A_atomic_weight'],df['B_atomic_weight'])
weigh_concat = pd.concat(weigh_comb)
df['A_atomic_weight'] = make_normalised_list(df['A_atomic_weight'],weigh_concat)
df['B_atomic_weight'] = make_normalised_list(df['B_atomic_weight'],weigh_concat)


# Atomic Radius rahm #
rahm_comb = (df['A_atomic_radius_rahm'],df['B_atomic_radius_rahm'])
rahm_concat = pd.concat(rahm_comb)
df['A_atomic_radius_rahm'] = make_normalised_list(df['A_atomic_radius_rahm'],rahm_concat)
df['B_atomic_radius_rahm'] = make_normalised_list(df['B_atomic_radius_rahm'],rahm_concat)


# Mendeleev Number #
mend_comb = (df['A_mendeleev_number'],df['B_mendeleev_number'])
mend_concat = pd.concat(mend_comb)
df['A_mendeleev_number'] = make_normalised_list(df['A_mendeleev_number'],mend_concat)
df['B_mendeleev_number'] = make_normalised_list(df['B_mendeleev_number'],mend_concat)


# Pettifor Number #
pett_comb = (df['A_pettifor_number'],df['B_pettifor_number'])
pett_concat = pd.concat(pett_comb)
df['A_pettifor_number'] = make_normalised_list(df['A_pettifor_number'],pett_concat)
df['B_pettifor_number'] = make_normalised_list(df['B_pettifor_number'],pett_concat)


# Glawe Number #
glaw_comb = (df['A_glawe_number'],df['B_glawe_number'])
glaw_concat = pd.concat(glaw_comb)
df['A_glawe_number'] = make_normalised_list(df['A_glawe_number'],glaw_concat)
df['B_glawe_number'] = make_normalised_list(df['B_glawe_number'],glaw_concat)

# Abundance Crust #
ab_comb = (df['A_abundance_crust'],df['B_abundance_crust'])
ab_concat = pd.concat(glaw_comb)
df['A_abundance_crust'] = make_normalised_list(df['A_abundance_crust'],ab_concat)
df['B_abundance_crust'] = make_normalised_list(df['B_abundance_crust'],ab_concat)

df.to_csv('test.csv')



## -----------Generating Feature Correlation ------------------------------ ##

parameters = df.columns[:len(df.columns)].values.tolist()

len_orig_params = len(parameters)


df1 = df.iloc[:, 26:32]


corrMatrix = df.corr()

df1_parameters = df1.columns[:len(df1.columns)].values.tolist()

params_reduced = []

for i in df1_parameters:
    new1 = i.replace('_',' ')
    params_reduced.append(new1) 


# f = plt.figure(3)
# plt.matshow(corrMatrix, fignum=f.number)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.xticks(range(len(params_reduced)), params_reduced, rotation=45)
# plt.yticks(range(len(params_reduced)), params_reduced)
# cb = plt.colorbar()
# #cb.ax.tick_params(labelsize=10)
# #plt.title('Correlation Matrix', fontsize=16);
# plt.show()

# f = plt.figure(3)
# plt.matshow(corrMatrix, fignum=f.number)
# plt.xticks([], [])
# plt.yticks([], [])
# # plt.xticks(range(len(params_reduced)), params_reduced, rotation=45)
# # plt.yticks(range(len(params_reduced)), params_reduced)
# cb = plt.colorbar()
# #cb.ax.tick_params(labelsize=10)
# #plt.title('Correlation Matrix', fontsize=16);
# plt.show()


 


  
   
corr_lst = make_corr_list(corrMatrix, parameters, 0.95)

no_laps = len(corr_lst)

neg_corr_lst = make_neg_corr_list(corrMatrix, parameters, -0.85)








params_to_remove_c = [ 'A_atomic_number','B_atomic_number',
                    'A_pettifor_number','B_pettifor_number', 
                    'A_melting_point', 'B_melting_point',
                    'A_boiling_point', 'B_boiling_point',
                     'A_covalent_radius_pyykko_double', 'B_covalent_radius_pyykko_double',
                      'A_covalent_radius_cordero','B_covalent_radius_cordero',
                      'A_covalent_radius_pyykko', 'B_covalent_radius_pyykko',
                       'A_covalent_radius_slater','B_covalent_radius_slater',
                        'A_mendeleev_number','B_mendeleev_number',
                         'A_vdw_radius_mm3','B_vdw_radius_mm3']

for i in params_to_remove_c:
    parameters.remove(i)


parameters.remove('Band gap [eV]')

print('\n{} features removed from feature correlation, leaving {} left'.format(len(params_to_remove_c),len(parameters)))






## ------ Implementing the RF algorithm ------- ##

tolerance = 1
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75

# Define label. Last argument is True if just predicting zero or non-zero #
df['label']=discretize(df['Band gap [eV]'].values.tolist(),tolerance, True)


train,test = df[df['is_train']==True],df[df['is_train']==False]

perc_zeros_old = test['label'].values.tolist().count('zero')/len(test['label'])
print('% zeros in unfiltered testing data = {}'.format(round(perc_zeros_old, 2)))


# control the number of zero band gap samples in the testing data 
test = delete_label_zeros(test,test['label'],0.65)
perc_zeros = test['label'].values.tolist().count('zero')/len(test['label'])


labels = np.unique(train['label'])
labels_numeric = make_numeric(train['label'])
label_table = make_ref_table(df['label'])


# Define x and y training and testing samples
X_train = train[parameters]
y_train = make_numeric(train['label'])

X_test = test[parameters]
y_test = make_numeric(test['label'])

## Define Random Forest Classifier ##
clf = RandomForestClassifier(n_jobs=None, random_state=None)
clf.fit(X_train,y_train)


# Random Forest Predictions #
preds_numeric = clf.predict(X_test)
preds = labels[preds_numeric]

compare = make_comp_table(test['label'],preds)
acc = find_accuracy(test['label'],preds)


print('% zeros in testing data = {}\n'.format(round(perc_zeros,2)))
print('Accuracy = {}'.format(acc))
print('No. samples in training data = {}'.format(len(train)))
print('No. samples in testing data = {}\n'.format(len(test)))




features = ['non-zero', 'zero']

from data_distribution import *

## use for discretised bins classification
# cf_matrix = confusion_matrix(y_test,preds_numeric)
# ax = plt.subplot()
# sn.heatmap(cf_matrix, annot=True, ax = ax, fmt = 'g', cmap = 'Blues', cbar = True)
# ax.set_xticklabels(labels = cl, rotation=45)
# ax.set_yticklabels(labels = cl, rotation=0)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')



## use for binary classification
cf_matrix = confusion_matrix(y_test,preds_numeric)
ax = plt.subplot()
sn.heatmap(cf_matrix, annot=True, ax = ax, fmt = 'g', cmap = 'Blues', cbar = False)
ax.set_xticklabels(labels = features, rotation=45)
ax.set_yticklabels(labels = features, rotation=0)
plt.ylabel('Actual')
plt.xlabel('Predicted')



## ------------------------ Feature Removal from Importance ------------------------------- ##


performance = clf.feature_importances_

# outputs the indices as per the sorted performance #
indices = np.argsort(performance)

# arranges the parameters as per the performance
arr_params = [parameters[i] for i in indices]

# make an arranged parameters list without underscores   
params_graphing = []
for i in arr_params:
    new1 = i.replace('_',' ')
    params_graphing.append(new1)        


y_pos = np.arange(len(parameters))

p = plt.figure(2)
plt.barh(y_pos, performance[indices],color="g", align="center")
plt.yticks(y_pos, params_graphing, rotation = 0)
plt.xlabel('Performance')
plt.title('Feature Importance')


# We will use this list to cull features 
ranked = arr_params[::-1]
best_features  = ranked[:5]
worst_features = ranked[-25:]
perf_sorted = sorted(performance, reverse = True)


# Count how many performances are below certian value
how_many = how_many_above(0.019, perf_sorted)

    
# Generate a list of parameters to be removed in SVR model
params_to_remove_fs = ranked[how_many:]  


print('No. features to be removed from FS is {}, from a total of {}'.format(len(params_to_remove_fs),len_orig_params))
## ------ Feature Removal Ends ---------------- ## 



















