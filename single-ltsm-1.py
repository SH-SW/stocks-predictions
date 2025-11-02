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

dataset_train_arr = np.hstack((X_train, y_train))
dataset_test_arr = np.hstack((X_test, y_test))

SEQ_LEN = 50

# -----------------------
# Dataset
# -----------------------
class MyDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window, :-1]    # (window, features)
        y = self.data[idx + self.window, -1]          # scalar target
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = MyDataset(dataset_train_arr, SEQ_LEN)
test_dataset = MyDataset(dataset_test_arr, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -----------------------
# Model
# -----------------------
class StockLSTM(nn.Module):
    def __init__(self, input_dims, hidden_size=32, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_dims)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden)
        out = out[:, -1, :]             # last timestep -> (batch, hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)             # (batch, 1)
        return out

# -----------------------
# Train / test utilities
# -----------------------
def train_test_model(model, criterion, optimizer, train_loader, test_loader, epochs=200, tolerance=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_history = []

    for epoch in range(1, epochs+1):
        model.train()
        batch_losses = []
        for features, labels in train_loader:
            features = features.to(device)           # (batch, seq, feat)
            labels = labels.to(device).float()       # (batch,)
            outputs = model(features).squeeze(-1)    # (batch,)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.item())

        loss_history.extend(batch_losses)
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - train loss: {np.mean(batch_losses):.6f}")

    # save model
    os.makedirs(results_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(results_path, "model.ckpt"))

    # evaluate on test set
    model.eval()
    correct = 0
    total = 0
    preds = []
    trues = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device).float()
            outputs = model(features).squeeze(-1)   # (batch,)
            within = torch.abs(outputs - labels) <= tolerance
            correct += int(within.sum().item())
            total += labels.numel()

            preds.extend(outputs.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test accuracy (within tolerance={tolerance}): {acc:.2f}% ({correct}/{total})")

    # inverse transform to price scale for readability
    preds_orig = scalers['Close'].inverse_transform(np.array(preds).reshape(-1,1)).flatten() if preds else np.array([])
    trues_orig = scalers['Close'].inverse_transform(np.array(trues).reshape(-1,1)).flatten() if trues else np.array([])

    plt.figure(figsize=(10,5))
    plt.plot(preds_orig, label='Predictions (orig scale)')
    plt.plot(trues_orig, label='Actual (orig scale)')
    plt.legend()
    plt.title('Predictions vs Actual (original price scale)')
    plt.show()

    return loss_history

def predict(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features).squeeze(-1)
            preds.extend(outputs.cpu().numpy().tolist())
            trues.extend(labels.numpy().tolist())

    preds_orig = scalers['Close'].inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    trues_orig = scalers['Close'].inverse_transform(np.array(trues).reshape(-1,1)).flatten()
    return preds, trues, preds_orig, trues_orig

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    model = StockLSTM(input_dims=4, hidden_size=32, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_hist = train_test_model(model, criterion, optimizer, train_loader, test_loader, epochs=200, tolerance=0.05)
    preds, trues, preds_orig, trues_orig = predict(model, test_loader)
    # optional: print first 5 predictions in original scale
    for i in range(min(5, len(preds_orig))):
        print(f"pred={preds_orig[i]:.2f}, actual={trues_orig[i]:.2f}")