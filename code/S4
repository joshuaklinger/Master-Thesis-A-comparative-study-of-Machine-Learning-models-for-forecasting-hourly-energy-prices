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
from fastdtw import fastdtw
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import math
from math import sqrt

# Import S4Model from s4torch
from s4torch.model import S4Model

# Configuration
HISTORY_LEN = 168  # 96 hours of history
PRED_LEN = 96     # 48 hours of prediction
BATCH_SIZE = 64
EPOCHS_PER_RUN = 1000
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join("checkpoints", "S4_final.pt")
print(f"Using device: {DEVICE}")

class PriceDataset(Dataset):
    def __init__(self, data, timestamps, history_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.timestamps = timestamps
        self.history_len = history_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.history_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.history_len].unsqueeze(-1)
        y = self.data[idx + self.history_len: idx + self.history_len + self.pred_len].unsqueeze(-1)
        return x, y

def plot_test_sample(x, y_true, y_pred, timestamps, scaler, title="Test Sample", save_path=None):
    plt.figure(figsize=(15, 6))
    
    # Inverse scaling
    x_inv = scaler.inverse_transform(x.cpu().numpy().reshape(-1, 1)).flatten()
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

class S4Predictor(nn.Module):
    def __init__(self, d_input=1, d_model=64, d_output=PRED_LEN, n_blocks=2, n=64, l_max=HISTORY_LEN):
        super(S4Predictor, self).__init__()
        self.s4 = S4Model(
            d_input=d_input,
            d_model=d_model,
            d_output=d_model,  # Changed from d_output to maintain dimensions
            n_blocks=n_blocks,
            n=n,
            l_max=l_max,
            wavelet_tform=False,
            collapse=False,
            p_dropout=0.1,
            activation=nn.GELU,
            norm_type='layer',
            norm_strategy='post',
            pooling=None
        )
        # Additional layers to process the S4 output
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(d_model * HISTORY_LEN, d_model)
        self.fc2 = nn.Linear(d_model, PRED_LEN)
        
    def forward(self, x):
        # Input shape: (batch, seq_len, 1)
        # S4Model expects (batch, seq_len, d_input)
        x = x.transpose(1, 2)  # (batch, 1, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, 1)
        
        # Process through S4
        out = self.s4(x)  # Shape: (batch, seq_len, d_model)
        
        # Flatten and predict
        out = self.flatten(out)  # (batch, seq_len * d_model)
        out = self.fc1(out)  # (batch, d_model)
        out = self.fc2(out)  # (batch, PRED_LEN)
        
        return out.unsqueeze(-1)  # (batch, PRED_LEN, 1)

def load_data():
    df = pd.read_csv("./dataset/all_countries.csv")
    df = df[df["ISO3 Code"] == "DEU"].copy()
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.sort_values("Datetime (UTC)")
    df["Price (EUR/MWhe)"] = df["Price (EUR/MWhe)"].interpolate().bfill()
    
    # Extract raw prices and timestamps
    prices = df["Price (EUR/MWhe)"].values.reshape(-1, 1).astype(np.float32)
    timestamps = df["Datetime (UTC)"].values
    
    # First split into train/val/test (70%/20%/10%)
    total_samples = len(prices)
    train_end = int(total_samples * 0.7)
    val_end = train_end + int(total_samples * 0.2)
    
    train_prices = prices[:train_end]
    val_prices = prices[train_end:val_end]
    test_prices = prices[val_end:]
    
    # Scale data using only training data statistics
    scaler = MinMaxScaler()
    train_prices_scaled = scaler.fit_transform(train_prices).flatten()
    val_prices_scaled = scaler.transform(val_prices).flatten()
    test_prices_scaled = scaler.transform(test_prices).flatten()
    
    # Get corresponding timestamps
    train_timestamps = timestamps[:train_end]
    val_timestamps = timestamps[train_end:val_end]
    test_timestamps = timestamps[val_end:]
    
    return (train_prices_scaled, val_prices_scaled, test_prices_scaled,
            train_timestamps, val_timestamps, test_timestamps,
            scaler)

def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, loader, scaler, description="Validation"):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    metrics = {'mse': [], 'mae': [], 'rmse': [], 'dir_ac': []}
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc=description):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            
            # Convert to numpy and inverse scale
            y_np = y.cpu().numpy().reshape(-1, 1)
            pred_np = pred.cpu().numpy().reshape(-1, 1)
            y_inv = scaler.inverse_transform(y_np).flatten()
            pred_inv = scaler.inverse_transform(pred_np).flatten()

            true_diff = np.diff(y_inv)
            pred_diff = np.diff(pred_inv)
            dir_acc = np.mean(np.sign(true_diff) == np.sign(pred_diff))            
            
            # Calculate metrics
            metrics['mse'].append(mean_squared_error(y_inv, pred_inv))
            metrics['mae'].append(mean_absolute_error(y_inv, pred_inv))
            metrics['rmse'].append(sqrt(mean_squared_error(y_inv, pred_inv)))
            metrics['dir_ac'].append(dir_acc)
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    avg_metrics['loss'] = total_loss / len(loader)
    return avg_metrics

def print_metrics(metrics, description):
    print(f"\n{description} Metrics:")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"MSE:  {metrics['mse']:.4f}") 
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Directional Accuracy: {metrics['dir_ac']:.4f}")

def main():
    SEED =  42   #random.randint(0, 2**32 - 1)
    print(f"Random seed: {SEED}")
    torch.manual_seed(SEED)  
    np.random.seed(SEED)
    random.seed(SEED)

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data with proper scaling and splitting
    (train_data, val_data, test_data,
     train_timestamps, val_timestamps, test_timestamps,
     scaler) = load_data()
    
    # Create datasets and dataloaders
    train_dataset = PriceDataset(train_data, train_timestamps, HISTORY_LEN, PRED_LEN)
    val_dataset = PriceDataset(val_data, val_timestamps, HISTORY_LEN, PRED_LEN)
    test_dataset = PriceDataset(test_data, test_timestamps, HISTORY_LEN, PRED_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # Initialize S4 model
    model = S4Predictor(
        d_input=1,
        d_model=56, # 40
        d_output=56, # 40
        n_blocks=2,   # 2
        n=56, # 40
        l_max=HISTORY_LEN
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Early stopping parameters
    patience = 50
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Training loop with validation and early stopping
    for epoch in range(EPOCHS_PER_RUN):
        print(f"\nTraining epoch {epoch+1}/{EPOCHS_PER_RUN}")
        train_loss = train_model(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}")

        # Validate every epoch
        val_metrics = evaluate_model(model, val_loader, scaler, "Validation")
        print_metrics(val_metrics, "Validation")

        # Check for improvement
        current_val_loss = val_metrics['loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, CHECKPOINT_PATH)
            print(f"Validation loss improved. Model saved to {CHECKPOINT_PATH}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve}/{patience} epochs")
            
            # Early stopping check
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                early_stop = True
                break

        # Save checkpoint every 5 epochs regardless
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, f"checkpoints/epoch_{epoch+1}.pt")
            print(f"Checkpoint saved for epoch {epoch+1}")

    if not early_stop:
        print("\nTraining completed without early stopping")

    # Load the best model before final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Best model loaded (from epoch {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.6f})")

    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_loader, scaler, "Testing")
    print_metrics(test_metrics, "Test")

    # Plot one test sample
    test_idx = random.randint(0, len(test_dataset) - 1)
    x, y = test_dataset[test_idx]
    x = x.unsqueeze(0).to(DEVICE)
    y = y.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred = model(x)
    
    # Get corresponding timestamps
    sample_timestamps = test_timestamps[test_idx:test_idx + HISTORY_LEN + PRED_LEN]
    plot_test_sample(x[0], y[0], pred[0], sample_timestamps, scaler, 
                   title="Test Sample - Best Model", 
                   save_path="s4_best.png")

if __name__ == "__main__":
    main()
