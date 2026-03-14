import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
device = 'cpu'
import torchmetrics

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib

from src.model import MyModel

# ---------- Load ----------
df = pd.read_csv("data/diabetes.csv")

X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]


# ---------- Preprocessing ----------
num_cols = ["Age", "BMI", "Family_History_of_Diabetes"]

cat_cols = [
    "Sex",
    "Ethnicity",
    "Alcohol_Consumption",
    "Smoking_Status"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X = preprocessor.fit_transform(X)
joblib.dump(preprocessor, "preprocessor.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

# ---------- DataLoader (Mini Batch GD) ----------
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = 32)



def evaluate(model, data_loader, metric) :
  model.eval()
  metric.reset()
  with torch.no_grad() :
    for X_batch, y_batch in data_loader :
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      y_pred = model(X_batch)
      metric.update(y_pred,y_batch)
  return metric.compute()

def train(model,optimizer, loss_fun , metric, train_loader, valid_loader,n_epoch,
          patience=2, factor=0.5, epoch_callback=None) :

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=factor)
  history = {'epoch':[],'training loss':[],'train metric':[],'valid metric':[]}
  for i in range(n_epoch) :
    total_loss = 0
    metric.reset()
    model.train()
    if epoch_callback is not None:
            epoch_callback(model, i)
    for X_batch, y_batch in train_loader :
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      y_pred = model(X_batch)
      temp_loss = loss_fun(y_pred, y_batch)
      total_loss += temp_loss.item()
      temp_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      metric.update(y_pred, y_batch)

    history['epoch'].append(i+1)
    history['training loss'].append(total_loss/len(train_loader))
    history['train metric'].append(metric.compute().item())
    val_metric = evaluate(model,valid_loader,metric).item()
    history['valid metric'].append(val_metric)
    scheduler.step(val_metric)

    if (i+1)%1 == 0 :
      print(f'epoch {i+1}/{n_epoch} ',
            f'training loss: {history['training loss'][-1]:.4f} ',
            f'train metric: {history['train metric'][-1]:.4f} ',
            f'velid metric: {history['valid metric'][-1]:.4f}')
  return history



model = MyModel(input_dim=X_train.shape[1])  #15
n_epoch = 5
loss_fun = nn.BCELoss()
metric = torchmetrics.Accuracy(task='binary')
optimizer = torch.optim.NAdam(model.parameters())
history = train(model, optimizer, loss_fun, metric, train_loader, test_loader, n_epoch)

pd.DataFrame(history).to_csv("history.csv", index=False)
torch.save(model.state_dict(), "model.pth")
print("Final feature dimension:", X_train.shape[1])