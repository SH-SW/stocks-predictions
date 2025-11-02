import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# -----------------------
# Config
# -----------------------
CONFIG = {
    'input_dims': 5,  # 4 price features + 5 technical indicators
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 250,
    'patience': 15,
    'sequence_length': 50
}

# -----------------------
# Paths
# -----------------------
proj_root = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(proj_root, "data")
results_path = os.path.join(proj_root, "results")
csv_filename = "aapl.us.csv"

# -----------------------
# Technical Indicators
# -----------------------
def add_technical_indicators(df):
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    return df.fillna(0)

# -----------------------
# Helpers
# -----------------------
def str_to_datetime(s):
    y, m, d = map(int, s.split('-'))
    return datetime.datetime(year=y, month=m, day=d)

def df_to_X_y(df):
    arr = df.to_numpy()
    X = arr[:-1, :]           # features for day t
    Y = arr[1:, :4]          # predict all 4 price values
    return X.astype(np.float32), Y.astype(np.float32)

"""
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

"""

# -----------------------
# Dataset
# -----------------------
class ImprovedDataset(Dataset):
    def __init__(self, data, window):
        self.data = torch.FloatTensor(data)
        self.window = window

    def __getitem__(self, index):
        x = self.data[index:index + self.window, :-4]    # all features except last 4 (targets)
        y = self.data[index + self.window - 1, -4:]      # predict next day's OHLC
        return x, y

    def __len__(self):
        return len(self.data) - self.window

# -----------------------
# Model
# -----------------------
class ImprovedStockLSTM(nn.Module):
    def __init__(self, input_dims, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        
        self.lstm1 = nn.LSTM(input_dims, hidden_size, num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers, 
                            dropout=dropout, batch_first=True)
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 4)  # predict OHLC
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # Bidirectional LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Attention
        x_permuted = x.permute(1, 0, 2)
        attn_out, _ = self.attention(x_permuted, x_permuted, x_permuted)
        x = attn_out[-1]
        
        # Dense layers with residual
        identity = x
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity
        
        x = self.fc2(x)
        return x

# -----------------------
# Training
# -----------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    #early_stopping = EarlyStopping(patience=config['patience'])
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_losses = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        #early_stopping(val_loss)
        #if early_stopping.early_stop:
        #    print("Early stopping triggered")
        #    break
    
    return history

# -----------------------
# Main execution
# -----------------------
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv(os.path.join(data_path, csv_filename))
    df['Date'] = df['Date'].apply(str_to_datetime)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    df.index = df.pop('Date')
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Train/val/test split
    train_size = int(len(df_scaled) * 0.7)
    val_size = int(len(df_scaled) * 0.15)
    
    train_data = df_scaled.iloc[:train_size]
    val_data = df_scaled.iloc[train_size:train_size+val_size]
    test_data = df_scaled.iloc[train_size+val_size:]
    
    # Create datasets
    train_dataset = ImprovedDataset(train_data.values, CONFIG['sequence_length'])
    val_dataset = ImprovedDataset(val_data.values, CONFIG['sequence_length'])
    test_dataset = ImprovedDataset(test_data.values, CONFIG['sequence_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize model
    model = ImprovedStockLSTM(
        input_dims=CONFIG['input_dims'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train model
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        CONFIG
    )
    
    # Save model
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': CONFIG,
        'scaler': scaler
    }, os.path.join(results_path, 'improved_model.pt'))
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()