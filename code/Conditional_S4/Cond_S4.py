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
from s4torch.model import S4Model
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
HISTORY_LEN = 168  # 168 hours of history
PRED_LEN = 96      # 96 hours of prediction
BATCH_SIZE = 64
EPOCHS = 1000      # Set a large number, as early stopping will handle it
INIT_LR = 1e-4     # Initial learning rate
MIN_LR = 1e-6      # Minimum learning rate
PATIENCE = 80      # Early stopping patience
GRADIENT_CLIP = 1.0 # Gradient clipping to prevent exploding gradients
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join("checkpoints", "test__seed_S4_weather_optimized.pt")
print(f"Using device: {DEVICE}")

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
        
        return x, y

def create_time_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['Datetime (UTC)'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Datetime (UTC)'].dt.hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['Datetime (UTC)'].dt.dayofweek / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['Datetime (UTC)'].dt.dayofweek / 7)
    
    return df[['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']].values

def load_data():
    df = pd.read_csv("./dataset/all_countries.csv")
    df = df[df["ISO3 Code"] == "DEU"].copy()
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.sort_values("Datetime (UTC)")
    df["Price (EUR/MWhe)"] = df["Price (EUR/MWhe)"].interpolate().bfill()
    
    try:
        weather_df = pd.read_csv("./dataset.txt", parse_dates=['Datetime (UTC)'])
        weather_df = weather_df.sort_values('Datetime (UTC)')
        df = pd.merge_asof(df, weather_df, on='Datetime (UTC)')
    except:
        print("Weather data not found, using only time features")
        weather_cols = []
    
    weather_features = [
        'TT_TU',
        'FF',
        'V_N',
        'SD_SO'
    ]
    
    time_features = create_time_features(df)
    prices = df["Price (EUR/MWhe)"].values.reshape(-1, 1).astype(np.float32)
    
    weather_data = np.zeros((len(df), len(weather_features)))
    for i, col in enumerate(weather_features):
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').bfill()
            weather_data[:, i] = df[col].values
        else:
            print(f"Warning: Weather feature {col} not found")
    
    timestamps = df["Datetime (UTC)"].values
    total_samples = len(prices)
    train_end = int(total_samples * 0.7)
    val_end = train_end + int(total_samples * 0.2)
    
    train_prices = prices[:train_end]
    val_prices = prices[train_end:val_end]
    test_prices = prices[val_end:]
    
    train_weather = weather_data[:train_end]
    val_weather = weather_data[train_end:val_end]
    test_weather = weather_data[val_end:]
    
    train_time = time_features[:train_end]
    val_time = time_features[train_end:val_end]
    test_time = time_features[val_end:]
    
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

class S4Predictor(nn.Module):
    def __init__(self, d_input=9, d_model=96, d_output=PRED_LEN, n_blocks=4, n=96, l_max=HISTORY_LEN):
        super(S4Predictor, self).__init__()
        self.s4 = S4Model(
            d_input=d_input,
            d_model=d_model,
            d_output=d_model,
            n_blocks=n_blocks,
            n=n,
            l_max=l_max,
            wavelet_tform=False,
            collapse=False,
            p_dropout=0.2,  # Increased dropout to 0.2
            activation=nn.GELU,
            norm_type='layer',
            norm_strategy='post',
            pooling=None
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(d_model * HISTORY_LEN, d_model)
        self.fc2 = nn.Linear(d_model, PRED_LEN)
        
    def forward(self, x):
        out = self.s4(x)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out.unsqueeze(-1)

def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, loader, price_scaler, description="Validation"):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc=description):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(pred.cpu().numpy())
            
    all_y_true = np.concatenate(all_y_true).reshape(-1, 1)
    all_y_pred = np.concatenate(all_y_pred).reshape(-1, 1)
    
    y_inv = price_scaler.inverse_transform(all_y_true).flatten()
    pred_inv = price_scaler.inverse_transform(all_y_pred).flatten()

    true_diff = np.diff(y_inv)
    pred_diff = np.diff(pred_inv)
    dir_ac = np.mean(np.sign(true_diff) == np.sign(pred_diff))
    
    metrics = {
        'loss': total_loss / len(loader),
        'mse': mean_squared_error(y_inv, pred_inv),
        'mae': mean_absolute_error(y_inv, pred_inv),
        'rmse': sqrt(mean_squared_error(y_inv, pred_inv)),
        'dir_ac': dir_ac
    }
    
    return metrics

def print_metrics(metrics, description):
    print(f"\n{description} Metrics:")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Directional Accuracy:   {metrics['dir_ac']:.4f}")

def plot_test_sample(x, y_true, y_pred, timestamps, scaler, title="Test Sample", save_path=None):
    plt.figure(figsize=(15, 6))
    
    x_price = x[:, 0].cpu().numpy()
    y_true_price = y_true[:, 0].cpu().numpy()
    y_pred_price = y_pred[:, 0].cpu().numpy()
    
    x_inv = scaler.inverse_transform(x_price.reshape(-1, 1)).flatten()
    y_true_inv = scaler.inverse_transform(y_true_price.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_price.reshape(-1, 1)).flatten()
    
    history_timestamps = timestamps[:HISTORY_LEN]
    future_timestamps = timestamps[HISTORY_LEN:HISTORY_LEN+PRED_LEN]
    
    plt.plot(history_timestamps, x_inv, 'b-', label='History', linewidth=2)
    plt.plot([history_timestamps[-1]] + list(future_timestamps), 
             [x_inv[-1]] + list(y_true_inv), 
             'g-', label='True Future', linewidth=2)
    plt.plot([history_timestamps[-1]] + list(future_timestamps), 
             [x_inv[-1]] + list(y_pred_inv), 
             'r:', label='Predicted Future', linewidth=2)
    
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

def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed: {SEED}")

    (train_prices, val_prices, test_prices,
     train_weather, val_weather, test_weather,
     train_time, val_time, test_time,
     train_timestamps, val_timestamps, test_timestamps,
     price_scaler) = load_data()
    
    train_dataset = PriceDataset(train_prices, train_weather, train_time, 
                                train_timestamps, HISTORY_LEN, PRED_LEN)
    val_dataset = PriceDataset(val_prices, val_weather, val_time,
                              val_timestamps, HISTORY_LEN, PRED_LEN)
    test_dataset = PriceDataset(test_prices, test_weather, test_time,
                               test_timestamps, HISTORY_LEN, PRED_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    sample_x, _ = train_dataset[0]
    input_size = sample_x.shape[-1]
    print(f"\nInput size: {input_size} features (1 price + {input_size-1} weather/time)")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    model = S4Predictor(
        d_input=input_size,
        d_model=36, # Increased model dimension
        d_output=36,
        n_blocks=2,
        n=36,     # Increased state dimension
        l_max=HISTORY_LEN
    ).to(DEVICE)
    print(f"\nModel architecture:")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-4)    # war 1e-4
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=MIN_LR) # Increased scheduler patience 0.1 und 15
    
    best_val_loss = float('inf')
    no_improve_count = 0
    
    #for epoch in range(EPOCHS):
        #print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        #train_loss = train_model(model, train_loader, optimizer, criterion)
        #print(f"Train Loss: {train_loss:.6f}")
        
        #val_metrics = evaluate_model(model, val_loader, price_scaler)
        #print_metrics(val_metrics, "Validation")
        
        #scheduler.step(val_metrics['loss'])
        
        #if val_metrics['loss'] < best_val_loss:
            #best_val_loss = val_metrics['loss']
            #no_improve_count = 0
            #torch.save({
                #'epoch': epoch,
                #'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                #'val_loss': best_val_loss,
            #}, CHECKPOINT_PATH)
            #print(f"Saved best model (val_loss={best_val_loss:.6f})")
        #else:
            #no_improve_count += 1
            #print(f"No improvement for {no_improve_count}/{PATIENCE} epochs.")
            #if no_improve_count >= PATIENCE:
                #print(f"Early stopping at epoch {epoch+1}.")
                #break
    
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\n--- Final Evaluation with Best Model ---")
    
    test_metrics = evaluate_model(model, test_loader, price_scaler, "Testing")
    print_metrics(test_metrics, "Final Test")
    
    test_idx = random.randint(0, len(test_dataset) - 1)
    x, y = test_dataset[test_idx]
    x = x.unsqueeze(0).to(DEVICE)
    y = y.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred = model(x)
    
    sample_timestamps = test_timestamps[test_idx:test_idx + HISTORY_LEN + PRED_LEN]
    plot_test_sample(x[0], y[0], pred[0], sample_timestamps, price_scaler, 
                     title="Conditioned S4", 
                     save_path="test_seed_s4_weather_best.png")
    
    print("\nExpected unconditional performance:")
    print("MSE: ~3000, RMSE: ~55, R2: ~0.8")
    print("\nWeather-enhanced S4 performance:")
    print(f"MSE: {test_metrics['mse']:.1f}, RMSE: {test_metrics['rmse']:.1f}, Directional Accuracy: {test_metrics['dir_ac']:.3f}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
