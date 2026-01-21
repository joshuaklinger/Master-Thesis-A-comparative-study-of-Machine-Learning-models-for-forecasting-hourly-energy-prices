import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import math
from math import sqrt

# Configuration (same as unconditional)
HISTORY_LEN = 168  # 168 hours of history
PRED_LEN = 96      # 96 hours of prediction
BATCH_SIZE = 64
EPOCHS_PER_RUN = 300
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join("checkpoints", "GRU_weather_optimized.pt")
print(f"Using device: {DEVICE}")

def plot_test_sample(x_price, y_true, y_pred, timestamps, scaler, title="Test Sample", save_path=None):
    plt.figure(figsize=(15, 6))

    # Inverse scaling
    x_inv = scaler.inverse_transform(x_price.cpu().numpy().reshape(-1, 1)).flatten()
    y_true_inv = scaler.inverse_transform(y_true.cpu().numpy().reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).flatten()

    # Get timestamps
    history_timestamps = timestamps[:HISTORY_LEN]
    future_timestamps = timestamps[HISTORY_LEN:HISTORY_LEN+PRED_LEN]

    # Plot history (blue)
    plt.plot(history_timestamps, x_inv, 'b-', label='History', linewidth=2)

    # Plot true future (green) - starting at the same point as history ends
    plt.plot([history_timestamps[-1]] + list(future_timestamps),
             [x_inv[-1]] + list(y_true_inv),
             'g-', label='True Future', linewidth=2)

    # Plot predicted future (red dotted) - starting at the same point as history ends
    plt.plot([history_timestamps[-1]] + list(future_timestamps),
             [x_inv[-1]] + list(y_pred_inv),
             'r:', label='Predicted Future', linewidth=2)

    # Add vertical line separator
    plt.axvline(x=history_timestamps[-1], color='gray', linestyle='--', linewidth=1)

    plt.title(title)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Price (EUR/MWh)')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

class PriceDataset(Dataset):
    def __init__(self, data, weather_data, time_features, timestamps, history_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.weather_data = torch.tensor(weather_data, dtype=torch.float32)
        self.time_features = torch.tensor(time_features, dtype=torch.float32)
        self.timestamps = timestamps
        self.history_len = history_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.history_len - self.pred_len
    
    def __getitem__(self, idx):
        # Price history (keep as primary feature)
        x_price = self.data[idx:idx + self.history_len].unsqueeze(-1)
        
        # Select only the most impactful weather features based on paper
        # Temperature, Wind Speed, Cloud Cover, Sunshine Duration
        x_weather = self.weather_data[idx:idx + self.history_len, :4]  # First 4 features
        
        # Only use most important time features
        x_time = self.time_features[idx:idx + self.history_len, :4]  # hour and weekday cyclical
        
        # Combine all features (price first)
        x = torch.cat([x_price, x_weather, x_time], dim=-1)
        
        # Target (future prices)
        y = self.data[idx + self.history_len: idx + self.history_len + self.pred_len].unsqueeze(-1)
        
        # Return the original price history separately for plotting
        return x, y, x_price

def create_time_features(df):
    # Create only the most essential cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df['Datetime (UTC)'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Datetime (UTC)'].dt.hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['Datetime (UTC)'].dt.dayofweek / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['Datetime (UTC)'].dt.dayofweek / 7)
    
    return df[['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']].values

def load_data():
    # Load price data
    df = pd.read_csv("./dataset/all_countries.csv")
    df = df[df["ISO3 Code"] == "DEU"].copy()
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.sort_values("Datetime (UTC)")
    df["Price (EUR/MWhe)"] = df["Price (EUR/MWhe)"].interpolate().bfill()
    
    # Load and merge weather data
    try:
        weather_df = pd.read_csv("./dataset.txt", parse_dates=['Datetime (UTC)'])
        weather_df = weather_df.sort_values('Datetime (UTC)')
        df = pd.merge_asof(df, weather_df, on='Datetime (UTC)')
    except:
        print("Weather data not found, using only time features")
        weather_cols = []
    
    # Select only the most important weather features from paper
    weather_features = [
        'TT_TU',  # Temperature (most important per paper)
        'FF',     # Wind speed (key for wind power)
        'V_N',    # Cloud cover (affects solar)
        'SD_SO'   # Sunshine duration (direct solar impact)
    ]
    
    # Create time features
    time_features = create_time_features(df)
    
    # Extract data
    prices = df["Price (EUR/MWhe)"].values.reshape(-1, 1).astype(np.float32)
    
    # Handle missing weather data
    weather_data = np.zeros((len(df), len(weather_features)))
    for i, col in enumerate(weather_features):
        if col in df.columns:
            weather_data[:, i] = df[col].values
        else:
            print(f"Warning: Weather feature {col} not found")
    
    timestamps = df["Datetime (UTC)"].values
    
    # Train/val/test split (70%/20%/10%)
    total_samples = len(prices)
    train_end = int(total_samples * 0.7)
    val_end = train_end + int(total_samples * 0.2)
    
    # Split data
    train_prices = prices[:train_end]
    val_prices = prices[train_end:val_end]
    test_prices = prices[val_end:]
    
    train_weather = weather_data[:train_end]
    val_weather = weather_data[train_end:val_end]
    test_weather = weather_data[val_end:]
    
    train_time = time_features[:train_end]
    val_time = time_features[train_end:val_end]
    test_time = time_features[val_end:]
    
    # Scale data
    price_scaler = MinMaxScaler()
    train_prices_scaled = price_scaler.fit_transform(train_prices).flatten()
    val_prices_scaled = price_scaler.transform(val_prices).flatten()
    test_prices_scaled = price_scaler.transform(test_prices).flatten()
    
    weather_scaler = StandardScaler()
    train_weather_scaled = weather_scaler.fit_transform(train_weather)
    val_weather_scaled = weather_scaler.transform(val_weather)
    test_weather_scaled = weather_scaler.transform(test_weather)
    
    time_scaler = StandardScaler()
    train_time_scaled = time_scaler.fit_transform(train_time)
    val_time_scaled = time_scaler.transform(val_time)
    test_time_scaled = time_scaler.transform(test_time)
    
    return (train_prices_scaled, val_prices_scaled, test_prices_scaled,
            train_weather_scaled, val_weather_scaled, test_weather_scaled,
            train_time_scaled, val_time_scaled, test_time_scaled,
            timestamps[:train_end], timestamps[train_end:val_end], timestamps[val_end:],
            price_scaler)

class SingleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(SingleGRU, self).__init__()
        # Same architecture as unconditional version
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out.unsqueeze(-1)

def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y, _ in tqdm(train_loader, desc="Training"): # Unpack the additional return value
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, loader, price_scaler, description="Validation"):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    metrics = {'mse': [], 'mae': [], 'rmse': [], 'dir_ac': []}
    
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc=description): # Unpack the additional return value
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            
            # Inverse scale only prices
            y_inv = price_scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1)).flatten()
            pred_inv = price_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten()

            true_diff = np.diff(y_inv)
            pred_diff = np.diff(pred_inv)
            dir_acc = np.mean(np.sign(true_diff) == np.sign(pred_diff))
            
            # Calculate metrics
            metrics['mse'].append(mean_squared_error(y_inv, pred_inv))
            metrics['mae'].append(mean_absolute_error(y_inv, pred_inv))
            metrics['rmse'].append(sqrt(mean_squared_error(y_inv, pred_inv)))
            metrics['dir_ac'].append(dir_acc)
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    avg_metrics['loss'] = total_loss / len(loader)
    return avg_metrics

def print_metrics(metrics, description):
    print(f"\n{description} Metrics:")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Directional Accuracy:    {metrics['dir_ac']:.4f}")

def main():
    # Fixed seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed: {SEED}")

    # Load data
    (train_prices, val_prices, test_prices,
     train_weather, val_weather, test_weather,
     train_time, val_time, test_time,
     train_timestamps, val_timestamps, test_timestamps,
     price_scaler) = load_data()
    
    # Create datasets
    
    class PriceDataset(Dataset):
        def __init__(self, data, weather_data, time_features, timestamps, history_len, pred_len):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.weather_data = torch.tensor(weather_data, dtype=torch.float32)
            self.time_features = torch.tensor(time_features, dtype=torch.float32)
            self.timestamps = timestamps
            self.history_len = history_len
            self.pred_len = pred_len

        def __len__(self):
            return len(self.data) - self.history_len - self.pred_len

        def __getitem__(self, idx):
            x_price = self.data[idx:idx + self.history_len].unsqueeze(-1)
            x_weather = self.weather_data[idx:idx + self.history_len, :4]
            x_time = self.time_features[idx:idx + self.history_len, :4]
            x = torch.cat([x_price, x_weather, x_time], dim=-1)
            y = self.data[idx + self.history_len: idx + self.history_len + self.pred_len].unsqueeze(-1)
            return x, y, x_price # <-- Modified to return x_price

    train_dataset = PriceDataset(train_prices, train_weather, train_time,
                                 train_timestamps, HISTORY_LEN, PRED_LEN)
    val_dataset = PriceDataset(val_prices, val_weather, val_time,
                               val_timestamps, HISTORY_LEN, PRED_LEN)
    test_dataset = PriceDataset(test_prices, test_weather, test_time,
                                test_timestamps, HISTORY_LEN, PRED_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    sample_x, _, _ = train_dataset[0] # Unpack the additional return value
    input_size = sample_x.shape[-1]
    print(f"\nInput size: {input_size} features (1 price + {input_size-1} weather/time)")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    model = SingleGRU(input_size=input_size, hidden_size=64, num_layers=2, pred_len=PRED_LEN).to(DEVICE)
    print(f"\nModel architecture:")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    patience = 60
    min_delta = 0.0001
    patience_counter = 0
    best_val_loss = float('inf')

    #for epoch in range(EPOCHS_PER_RUN):
        #print(f"\nEpoch {epoch+1}/{EPOCHS_PER_RUN}")
        
        #train_loss = train_model(model, train_loader, optimizer, criterion)
        #print(f"Train Loss: {train_loss:.6f}")
        
        #val_metrics = evaluate_model(model, val_loader, price_scaler)
        #print_metrics(val_metrics, "Validation")
        
        #if val_metrics['mse'] < best_val_loss - min_delta:
            #best_val_loss = val_metrics['mse']
            #patience_counter = 0
            #torch.save({
                #'epoch': epoch,
                #'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                #'val_loss': best_val_loss,
            #}, CHECKPOINT_PATH)
            #print(f"Saved best model (val_loss={best_val_loss:.6f})")
        #else:
            #patience_counter += 1
            #print(f"Patience counter: {patience_counter}/{patience}")

        #if patience_counter >= patience:
            #print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss for {patience} consecutive epochs.")
            #break

    print(f"\nLoading best model from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_loader, price_scaler, "Testing")
    print_metrics(test_metrics, "Final Test")
    
    print("\nExpected unconditional performance:")
    print("MSE: ~3000, RMSE: ~55, Directional Accuracy: ~0.8")
    print("\nWeather-enhanced performance:")
    print(f"MSE: {test_metrics['mse']:.1f}, RMSE: {test_metrics['rmse']:.1f}, Directional Accuracy: {test_metrics['dir_ac']:.3f}")

    print("\nPlotting a sample from the test set...")
    test_idx = random.randint(0, len(test_dataset) - 1)
    x, y, x_price = test_dataset[test_idx]
    x = x.unsqueeze(0).to(DEVICE)
    y = y.unsqueeze(0).to(DEVICE)
    x_price = x_price.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred = model(x)
        
    sample_timestamps = test_timestamps[test_idx:test_idx + HISTORY_LEN + PRED_LEN]
    plot_test_sample(x_price[0], y[0], pred[0], sample_timestamps, price_scaler,
                     title="Conditioned GRU",
                     save_path="GRU_weather_optimized_sample.png")

if __name__ == "__main__":
    main()
