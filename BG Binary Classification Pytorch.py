import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size = 15)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from elim_zeros_3 import delete_number_zeros


data = pd.read_csv('df_for_pytorch_binary.csv')
X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]

"""Preprocess, normalize and create the model"""
size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size)
X_test, y_test = delete_number_zeros(X_test,y_test, 0.65) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.values
y_test = y_test.values

class BinaryClassification(torch.nn.Module):
  def __init__(self, input_dimension):
    super().__init__()
    self.linear = torch.nn.Linear(input_dimension, 1)

  def forward(self, input_dimension):
      return self.linear(input_dimension)
    
_, input_dimension = X_train.shape

model = torch.nn.Linear(input_dimension, 1)

"""train the model"""

def configure_loss_function(): 
  return torch.nn.BCEWithLogitsLoss()

def configure_optimizer(model):
  return torch.optim.Adam(model.parameters())

def full_gd(model, criterion, optimizer, X_train, y_train, n_epochs=10000):
  train_losses = np.zeros(n_epochs)
  test_losses = np.zeros(n_epochs)

  for it in range(n_epochs): 
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs_test = model(X_test)
    loss_test = criterion(outputs_test, y_test)

    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()

    if (it + 1) % 50 == 0:
      print(f'In this epoch {it+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

  return train_losses, test_losses

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

criterion = configure_loss_function()
optimizer = configure_optimizer(model)
train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train)

"""evaluate model"""

with torch.no_grad():
  p_train = model(X_train)
  p_train = (p_train.numpy() > 0)

  train_acc = np.mean(y_train.numpy() == p_train)

  p_test = model(X_test)
  p_test = (p_test.numpy() > 0)
  
  test_acc = np.mean(y_test.numpy() == p_test)
  
print('Test Size: {}'.format(size))
print('Train Acc: {}'.format(train_acc))
print('Test Acc: {}'.format(test_acc))





features = ['non-zero', 'zero']




y_test_TF = (y_test > 0)

cf_matrix = confusion_matrix(y_test_TF,p_test)
ax = plt.subplot()
sn.heatmap(cf_matrix, annot=True, ax = ax, fmt = 'g', cmap = 'Blues')
ax.set_xticklabels(labels = features, rotation=45)
ax.set_yticklabels(labels = features, rotation=0)
plt.ylabel('Actual')
plt.xlabel('Predicted')


    
cmap = plt.cm.get_cmap('coolwarm')   

plt.figure(2)
plt.plot(train_losses, label = 'Train',color=cmap(0))
plt.plot(test_losses, label = 'Test', color = cmap(0.99))
plt.xlabel('Epoch no.')
plt.ylabel('Loss')
plt.legend()










