import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


from normalise import make_normalised_list, unnormalise, normalise

df = pd.read_csv("df_for_pytorch_cont.csv")




X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]


# Train - Test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, random_state=69)


# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=21)


scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)






y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)



class RegressionDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
    
train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())




EPOCHS = 800
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_FEATURES = len(X.columns)



train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 100)
        self.layer_2 = nn.Linear(100, 150)
        self.layer_out = nn.Linear(150, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        return (x)
    
    def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        return (x)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = MultipleRegression(NUM_FEATURES)
model.to(device)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_stats = {
    'train': [],
    "val": []
}



print("Begin training.")
for e in tqdm(range(1, EPOCHS+1)):
    
    # TRAINING
    train_epoch_loss = 0
    model.train()
    
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
            
            val_epoch_loss += val_loss.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))                              
        
       
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')


test_losses =  loss_stats['val']
train_losses = loss_stats['train']

cmap = plt.cm.get_cmap('coolwarm') 

plt.figure(1)
plt.plot(train_losses, label = 'Train', color=cmap(0))
plt.plot(test_losses, label = 'Test', color = cmap(0.99))
plt.xlabel('Epoch no.')
plt.ylabel('Loss')
plt.legend()

print('Number of samples used: {}'.format(len(df)))
print('Number of features used: {}'.format(NUM_FEATURES))

y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

mse = mean_squared_error(y_test, y_pred_list)
r_square = r2_score(y_test, y_pred_list)
print("Mean Squared Error :",mse)
print("R^2 :",r_square)


## Graphing ##    
bg_actual = y_test
bg_predict = y_pred_list 

    
    
difference = []
d_final = []
cmap_lst = []

zip_object = zip(bg_actual, bg_predict)
for list1_i, list2_i in zip_object:
    difference.append(abs(list1_i-list2_i))
    d = np.array(difference)
    d = normalise(d)

for norm_val in d :
    rev = 1 - norm_val        
    d_final.append(rev)

for d_val in d_final:
    colour = cmap(d_val)
    cmap_lst.append(colour)

x = np.linspace(0, 6, 100)   




bg_actual = y_test
bg_predict = y_pred_list 

plt.figure(2)
sc = plt.scatter(bg_actual, bg_predict, c=cmap_lst)
plt.plot(x,x,'--',color = 'black')
plt.xlabel('Actual Band Gap (eV)')
plt.ylabel('Predicted Band Gap (eV)')

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.05, 5.75, '$R^2$ = {}'.format(round(r_square,3)), fontsize=14,verticalalignment='top', bbox=props)

plt.show()








