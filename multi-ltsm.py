import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

# -----------------------
# Config / paths
# -----------------------
proj_root = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(proj_root, "data")
results_path = os.path.join(proj_root, "results")
csv_filename = "aapl.us.csv"  # adjust if different

# -----------------------
# Helpers
# -----------------------
def str_to_datetime(s):
    y, m, d = map(int, s.split('-'))
    return datetime.datetime(year=y, month=m, day=d)

def df_to_X_y(df):
    arr = df.to_numpy()
    X = arr[:-1, :]           # features for day t
    Y = arr[1:, -1]           # close price at day t+1 (last col)
    return X.astype(np.float32), Y.astype(np.float32)

# -----------------------
# Load & preprocess
# -----------------------
df = pd.read_csv(os.path.join(data_path, csv_filename))
df['Date'] = df['Date'].apply(str_to_datetime)
df = df[['Date', 'Open', 'High', 'Low', 'Close']]
df.index = df.pop('Date')

# scale each numeric column and keep scalers for inverse transforms
scalers = {}
for col in ['Open', 'High', 'Low', 'Close']:
    sc = MinMaxScaler(feature_range=(-1, 1))
    df[[col]] = sc.fit_transform(df[[col]])
    scalers[col] = sc

# train / test split
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx, :]
test = df.iloc[split_idx:, :]

X_train, y_train = df_to_X_y(train)
X_test, y_test = df_to_X_y(test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# plot scaled series (optional)
plt.plot(train["Close"], label='train')
plt.plot(test["Close"], label='test')
plt.legend()
plt.title("Scaled Close (train vs test)")
plt.show()

dataset_train = np.hstack((X_train, y_train))
dataset_test = np.hstack((X_test, y_test))
len_train = len(dataset_train)
len_test = len(dataset_test)
len_seq = 50

"""
structure of one element:

feature feature feature feature
feature feature feature feature
feature feature feature feature
target  target  target  target

using sliding windows
the columns are the open, high, low and close of the current day
by targeting at all the feature for the next day we can project data into the future
reuse outputs as inputs and capture trends for the future
"""

class MyDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = self.data[index:index + self.window, 0:4]  # keeping all features except the last column
        y = self.data[index+self.window, 4:]     # target all the features
        return x, y

    def __len__(self):
        return len(self.data) - self.window

dataset_train = MyDataset(dataset_train, len_seq)
dataset_test = MyDataset(dataset_test, len_seq)

data_loader_train = DataLoader(dataset_train, batch_size=len_seq)
data_loader_test = DataLoader(dataset_test, batch_size=1) # loading one sequence at a time
     

class StockLN(nn.Module):
    def __init__(self, input_dims, hidden_size, num_layers):
        super(StockLN, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_size, num_layers=num_layers, dropout=0.3, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 4) # for multilabel the output must be 4
        self.do = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[-1,-1,:]
        x = self.fc1(x)
        x = self.do(x)
        x = self.fc2(x)
        return x
     

def train_test_model(model, criterion, optimizer, train_loader, test_loader, epochs, tolerance):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_list = []
    for epoch in range(epochs):
        model.train()
        total_step = len(train_loader)
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(features)
            loss = criterion(outputs, labels[-1])

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_list.append(loss.item())
            if ((i+1) % 100 == 0) or (i==total_step-1):
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, epochs, i+1, total_step, loss.item()))

    # save the model checkpoint for transfer learning
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    torch.save(model.state_dict(), os.path.join(results_path, 'model.ckpt'))

    # testing
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            # network predictions
            outputs = model(features)

            within_tolerance = torch.abs(outputs - labels[-1]) <= tolerance

            if all(within_tolerance == True):
                correct+=1

        total = len_test
        test_accuracy = 100 * correct / total
        print('Final test Accuracy of the model: {:.2f} %'.format(test_accuracy))

    return loss_list
     

model = StockLN(input_dims=4, hidden_size=61, num_layers=1) 
criterion = nn.MSELoss(reduction="mean") # mean squared error loss for regression task
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_list = train_test_model(model, criterion, optimizer, data_loader_train, data_loader_test,epochs=250, tolerance=0.1)

def predict(model, num_days):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    current_data = df.iloc[int(len(df['Close']))-len_seq : , :].to_numpy().astype(np.float32)
    current_data = torch.tensor(current_data)
    current_data = current_data.unsqueeze(0).to(device)
    print(current_data.shape)

    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_days):
            prediction = model(current_data)
            predictions.append(prediction.cpu().numpy())
            prediction = prediction.unsqueeze(0).unsqueeze(0)
            current_data = torch.cat((current_data[:, -len_seq+1:, :], prediction), dim=1).to(device)

    return predictions
     

# generate predictions
number_of_days = 30 # one month is quite a lot
# this part will only caption the trend and not the fluctuations

predictions = predict(model, number_of_days)
print('Predictions:', predictions)

# starting point is adjusted using the test data
list1 = []
i = 0
for features, labels in data_loader_test:
    if i < number_of_days:
        list1.append(labels[-1])
        i += 1

# extracting the last element from each array in predictions
last_elements = [arr[-1] for arr in predictions]

# adjust the starting point of predictions to match the first element of the list
starting_point = list1[0]
prediction_start = last_elements[0]
offset = starting_point - prediction_start
adjusted_predictions = [pred + offset for pred in last_elements]

plt.figure(figsize=(8, 6))
plt.plot(list1, label='Actual')
plt.plot(adjusted_predictions, color='b', label='Predicted')
plt.title('Adjusted Predictions to Match Starting Point')
plt.xlabel('Array Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

def predict(model, train_loader, test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    with torch.no_grad():
        total = 0
        predictions = []
        labels_plot = []

        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            # network predictions
            outputs = model(features)

            predictions.append(outputs[-1].cpu().numpy())
            labels_plot.append(labels[-1][-1].cpu().numpy())

    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predictions', color='blue')
    plt.plot(labels_plot, label='Actual Labels', color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Model Predictions vs Actual Labels (Cost)')
    plt.legend()
    plt.show()

    return predictions
     

predictions = predict(model, data_loader_train ,data_loader_test)