import os
import time
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrizations
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt, log
import math

# Configuration - Tuned for maximum performance
HISTORY_LEN = 168 # 168 hours of history (7 days)
PRED_LEN = 96     # 96 hours of prediction (4 days)
BATCH_SIZE = 64
EPOCHS = 1000
INIT_LR = 5e-5    # Kept the lower learning rate for stability
MIN_LR = 1e-6
PATIENCE = 60     # Increased patience for a deeper model and more steps
GRADIENT_CLIP = 1.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.path.join("checkpoints", "beschte_diffwave_forecast.pt")
print(f"Using device: {DEVICE}")

# Diffusion parameters - Increased steps for higher accuracy
DIFFUSION_STEPS = 60 # 2484
DIFFUSION_SCHEDULE = 'linear'

class PriceDataset(Dataset):
    def __init__(self, data, timestamps, weather_data, history_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.timestamps = timestamps
        self.weather_data = torch.tensor(weather_data, dtype=torch.float32)
        self.history_len = history_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.history_len - self.pred_len

    def __getitem__(self, idx):
        # Concatenate history and future for both price and conditional data
        full_price_series = self.data[idx:idx + self.history_len + self.pred_len].unsqueeze(-1)
        
        cond_hist = self.weather_data[idx:idx + self.history_len]
        cond_fut = self.weather_data[idx + self.history_len: idx + self.history_len + self.pred_len]
        full_cond = torch.cat([cond_hist, cond_fut], dim=0)

        return full_price_series, full_cond

def plot_test_sample(full_series_true, y_pred, timestamps, scaler, title="Test Sample", save_path=None):
    plt.figure(figsize=(15, 6))
    
    # Inverse scaling for all data
    full_series_inv = scaler.inverse_transform(full_series_true.cpu().numpy().reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).flatten()

    # Split into history and future for plotting
    x_inv = full_series_inv[:HISTORY_LEN]
    y_true_inv = full_series_inv[HISTORY_LEN:]

    # Get timestamps
    history_timestamps = timestamps[:HISTORY_LEN]
    future_timestamps = timestamps[HISTORY_LEN:HISTORY_LEN+PRED_LEN]

    # Plot history (blue)
    plt.plot(history_timestamps, x_inv, 'b-', label='History', linewidth=2)

    # Plot true future (green)
    plt.plot([history_timestamps[-1]] + list(future_timestamps),
              [x_inv[-1]] + list(y_true_inv),
              'g-', label='True Future', linewidth=2)

    # Plot predicted future (red dotted)
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

    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.close()

class Diffusion(nn.Module):
    def __init__(self, steps=DIFFUSION_STEPS, schedule=DIFFUSION_SCHEDULE):
        super().__init__()
        self.steps = steps
        self.schedule = schedule

        if schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, steps, device=DEVICE)
        elif schedule == 'cosine':
            s = 0.008
            x = torch.linspace(0, steps, steps + 1, device=DEVICE)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_fn, x_0, t, noise=None, cond=None, loss_type="huber"):
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = denoise_fn(x_noisy, t, cond=cond)

        # We only calculate the loss on the prediction part of the series (the last PRED_LEN timesteps)
        pred_noise_slice = predicted_noise[:, -PRED_LEN:, :]
        true_noise_slice = noise[:, -PRED_LEN:, :]

        if loss_type == 'l1':
            loss = (true_noise_slice - pred_noise_slice).abs().mean()
        elif loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(true_noise_slice, pred_noise_slice)
        elif loss_type == "huber":
            loss = torch.nn.functional.smooth_l1_loss(true_noise_slice, pred_noise_slice)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, denoise_fn, x, t, t_index, cond=None):
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * denoise_fn(x, t, cond) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = (1 - self.alphas_cumprod[t-1]) / (1 - self.alphas_cumprod[t]) * self.betas[t]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, shape, cond=None):
        device = next(denoise_fn.parameters()).device
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(denoise_fn, x, t, i, cond)

        return x

    @torch.no_grad()
    def sample(self, denoise_fn, shape, cond=None):
        return self.p_sample_loop(denoise_fn, shape, cond)

def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=DEVICE) * -_embed)
    _embed = diffusion_steps.float().unsqueeze(1) * _embed.unsqueeze(0)
    _embed = torch.cat([torch.sin(_embed), torch.cos(_embed)], dim=-1)

    return _embed

def swish(x):
    return x * torch.sigmoid(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = parametrizations.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.out_channel = out_channel

    def forward(self, x):
        return self.conv(x)

class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation=1, diffusion_step_embed_dim_out=256):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)
        # Add dropout here
        self.dropout = nn.Dropout(p=0.1)

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = parametrizations.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = parametrizations.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        # Process diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.unsqueeze(-1)
        h = h + part_t

        h = self.dilated_conv_layer(h)
        # Apply dropout to the output of the dilated convolution
        h = self.dropout(h)
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip

class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers=20, dilation_cycle=10,
                  diffusion_step_embed_dim_in=128, diffusion_step_embed_dim_mid=256,
                  diffusion_step_embed_dim_out=256, cond_channels=None):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.skip_channels = skip_channels
        self.cond_channels = cond_channels

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        
        self.cond_proj = nn.Linear(self.cond_channels, res_channels) if cond_channels else None

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       dilation=2 ** (n % dilation_cycle),
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out))

    def forward(self, input_data):
        x, diffusion_steps, cond = input_data
        
        # Calculate diffusion step embedding
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = x
        
        if cond is not None and self.cond_proj is not None:
            cond_embed = self.cond_proj(cond)
            # Transpose the conditional embedding to match the shape of h
            cond_embed = cond_embed.transpose(1, 2)
            h = h + cond_embed
            
        skip = torch.zeros_like(x[:, :self.skip_channels, :])
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, diffusion_step_embed))
            skip = skip + skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)

class DiffWave(nn.Module):
    def __init__(self, input_channels=1, res_channels=40, skip_channels=40, out_channels=1,
                  num_res_layers=20, dilation_cycle=10,
                  diffusion_step_embed_dim_in=128,
                  diffusion_step_embed_dim_mid=256, diffusion_step_embed_dim_out=256,
                  cond_channels=None):
        super(DiffWave, self).__init__()

        self.init_conv = nn.Sequential(Conv(input_channels, res_channels, kernel_size=1), nn.ReLU())
        
        self.cond_channels = cond_channels

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             dilation_cycle=dilation_cycle,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             cond_channels=cond_channels)

        # Add dropout to the final convolution sequence
        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.1), # Dropout layer added here
            ZeroConv1d(skip_channels, out_channels)
        )

    def forward(self, x, diffusion_steps, cond=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        elif x.shape[-1] != 1:
            x = x[:, :, :1]

        x = x.transpose(1, 2)
        x = self.init_conv(x)
        
        cond_input = None
        if self.cond_channels:
            if cond is not None:
                cond_input = cond
            else:
                raise ValueError("Conditional data must be provided if cond_channels > 0")

        skip = self.residual_layer((x, diffusion_steps, cond_input))
        output = self.final_conv(skip)

        return output.transpose(1, 2)

def load_data():
    df = pd.read_csv("./dataset.txt")
    df = df[df["ISO3 Code"] == "DEU"].copy()
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.sort_values("Datetime (UTC)")

    # Interpolate and backfill prices
    df["Price (EUR/MWh)"] = df["Price (EUR/MWh)"].interpolate().bfill()
    prices = df["Price (EUR/MWh)"].values.reshape(-1, 1).astype(np.float32)

    # Define weather features
    weather_features = ['RF_TU', 'TT_TU', 'SD_SO', 'FF', 'DD', 'V_N', 'V_S1_CS', 'V_S1_CSA', 'V_S1_HHS',
                        'V_S1_NS', 'V_S2_CS', 'V_S2_CSA', 'V_S2_HHS', 'V_S2_NS', 'V_S3_CS',
                        'V_S3_CSA', 'V_S3_HHS', 'V_S3_NS', 'V_S4_CS', 'V_S4_HHS', 'V_S4_NS']
    
    # Handle missing weather data
    for col in weather_features:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').bfill()
        else:
            print(f"Warning: Weather feature '{col}' not found in DataFrame.")

    # Extract and scale price and weather data
    prices_scaler = MinMaxScaler()
    scaled_prices = prices_scaler.fit_transform(prices).flatten()

    weather_data = df[weather_features].values.astype(np.float32)
    weather_scaler = MinMaxScaler()
    scaled_weather = weather_scaler.fit_transform(weather_data)

    # Generate time-based features
    df['hour'] = df['Datetime (UTC)'].dt.hour
    df['day_of_week'] = df['Datetime (UTC)'].dt.dayofweek
    
    hour_one_hot = pd.get_dummies(df['hour'], prefix='hour').values.astype(np.float32)
    day_of_week_one_hot = pd.get_dummies(df['day_of_week'], prefix='day_of_week').values.astype(np.float32)
    
    # Combine weather and time features
    conditional_data = np.concatenate([scaled_weather, hour_one_hot, day_of_week_one_hot], axis=1)

    timestamps = df["Datetime (UTC)"].values

    return scaled_prices, timestamps, conditional_data, prices_scaler, conditional_data.shape[1]

def train_epoch(model, diffusion, train_loader, optimizer):
    model.train()
    total_loss = 0

    for full_price_series, full_cond in tqdm(train_loader, desc="Training"):
        full_price_series = full_price_series.to(DEVICE).float()
        full_cond = full_cond.to(DEVICE).float()
        
        t = torch.randint(0, diffusion.steps, (full_price_series.size(0),), device=DEVICE)

        optimizer.zero_grad()
        loss = diffusion.p_losses(model, full_price_series, t, cond=full_cond)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model, diffusion, loader, scaler, description="Validation"):
    model.eval()
    total_loss = 0
    all_y_true = []
    all_y_pred = []

    for full_price_series, full_cond in tqdm(loader, desc=description):
        full_price_series = full_price_series.to(DEVICE).float()
        full_cond = full_cond.to(DEVICE).float()

        batch_size = full_price_series.size(0)

        if len(full_price_series.shape) == 2:
            full_price_series = full_price_series.unsqueeze(-1)
            
        t = torch.randint(0, diffusion.steps, (batch_size,), device=DEVICE)
        loss = diffusion.p_losses(model, full_price_series, t, cond=full_cond)
        total_loss += loss.item()

        # --- Corrected and more robust sampling loop ---
        x_hist = full_price_series[:, :HISTORY_LEN, :]
        x = torch.randn(batch_size, HISTORY_LEN + PRED_LEN, 1, device=DEVICE)

        for i in reversed(range(0, diffusion.steps)):
            t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
            
            # For each denoising step, get a noisy version of the history to match training
            x_hist_noisy = diffusion.q_sample(x_0=x_hist, t=t)
            
            # Combine the noisy history with the current state of the denoising process for the future
            x_input = torch.cat([x_hist_noisy, x[:, HISTORY_LEN:, :]], dim=1)
            
            predicted_noise = model(x_input, t, cond=full_cond)
            
            # Denoise one step using the diffusion equation
            betas_t = diffusion.betas[t].view(-1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
            sqrt_recip_alphas_t = torch.sqrt(1.0 / diffusion.alphas[t]).view(-1, 1, 1)
            
            model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            
            if i == 0:
                x = model_mean
            else:
                posterior_variance_t = (1 - diffusion.alphas_cumprod[i-1]) / (1 - diffusion.alphas_cumprod[i]) * diffusion.betas[i]
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise

        pred = x[:, -PRED_LEN:, :]
        # --- End of corrected sampling loop ---
        
        all_y_true.append(full_price_series[:, -PRED_LEN:, :].cpu().numpy())
        all_y_pred.append(pred.cpu().numpy())

    # Calculate metrics
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    y_inv = scaler.inverse_transform(all_y_true.reshape(-1, 1)).flatten()
    pred_inv = scaler.inverse_transform(all_y_pred.reshape(-1, 1)).flatten()

    # Add directional accuracy
    y_diff = np.sign(y_inv[1:] - y_inv[:-1])
    pred_diff = np.sign(pred_inv[1:] - pred_inv[:-1])
    direction_acc = np.mean(y_diff == pred_diff)

    metrics = {
        'loss': total_loss / len(loader),
        'mse': mean_squared_error(y_inv, pred_inv),
        'mae': mean_absolute_error(y_inv, pred_inv),
        'rmse': sqrt(mean_squared_error(y_inv, pred_inv)),
        'direction_acc': direction_acc
    }

    return metrics

def print_metrics(metrics, description):
    print(f"\n{description} Metrics:")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Directional Accuracy: {metrics['direction_acc']:.4f}")

def main():
    # Set a fixed seed for reproducibility.
    SEED = 42
    print(f"Random seed: {SEED}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and prepare data
    price_series, timestamps, weather_data, price_scaler, COND_CHANNELS = load_data()

    total_samples = len(price_series)
    train_end = int(total_samples * 0.7)
    val_end = train_end + int(total_samples * 0.2)
    
    # Get the split conditional data
    train_weather_data = weather_data[:train_end]
    val_weather_data = weather_data[train_end:val_end]
    test_weather_data = weather_data[val_end:]

    train_dataset = PriceDataset(price_series[:train_end], timestamps[:train_end], train_weather_data, HISTORY_LEN, PRED_LEN)
    val_dataset = PriceDataset(price_series[train_end:val_end], timestamps[train_end:val_end], val_weather_data, HISTORY_LEN, PRED_LEN)
    test_dataset = PriceDataset(price_series[val_end:], timestamps[val_end:], test_weather_data, HISTORY_LEN, PRED_LEN)
    
    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Conditional features count: {COND_CHANNELS}")
    
    # Initialize model and diffusion - Optimized for a balance of performance and speed
    model = DiffWave(
        input_channels=1,
        res_channels=64,        # Increased from 40 for higher capacity
        skip_channels=64,       # Increased from 40 for higher capacity
        out_channels=1,
        num_res_layers=20,      # Increased from 20 for better temporal modeling
        dilation_cycle=10,
        diffusion_step_embed_dim_in=128,
        diffusion_step_embed_dim_mid=256,
        diffusion_step_embed_dim_out=256,
        cond_channels=COND_CHANNELS
    ).to(DEVICE)

    diffusion = Diffusion(steps=DIFFUSION_STEPS, schedule=DIFFUSION_SCHEDULE).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=MIN_LR)

    # Training loop with early stopping
    best_val_mse = float('inf')  # Corrected to track MSE
    no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_epoch(model, diffusion, train_loader, optimizer)
        print(f"Train Loss: {train_loss:.6f}")

        val_metrics = evaluate(model, diffusion, val_loader, price_scaler, "Validation")
        print_metrics(val_metrics, "Validation")

        scheduler.step(val_metrics['mse']) # Scheduler now tracks MSE

        # Early stopping check based on MSE
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Validation MSE improved to {best_val_mse:.4f}. Checkpoint saved.")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{PATIENCE} epochs.")
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model for final evaluation
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("\n--- Final Evaluation with Best Model ---")

    # Final evaluation
    test_metrics = evaluate(model, diffusion, test_loader, price_scaler, "Testing")
    print_metrics(test_metrics, "Test")

    # Generate and plot a sample prediction
    test_idx = random.randint(0, len(test_dataset) - 1)
    full_price_series, full_cond = test_dataset[test_idx]
    
    batch_size = 1
    full_price_series = full_price_series.unsqueeze(0).to(DEVICE)
    full_cond = full_cond.unsqueeze(0).to(DEVICE)

    # Re-use the corrected sampling loop from evaluate
    x_hist = full_price_series[:, :HISTORY_LEN, :]
    x = torch.randn(batch_size, HISTORY_LEN + PRED_LEN, 1, device=DEVICE)

    with torch.no_grad():
        for i in reversed(range(0, diffusion.steps)):
            t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
            
            x_hist_noisy = diffusion.q_sample(x_0=x_hist, t=t)
            
            x_input = torch.cat([x_hist_noisy, x[:, HISTORY_LEN:, :]], dim=1)
            
            predicted_noise = model(x_input, t, cond=full_cond)
            
            betas_t = diffusion.betas[t].view(-1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
            sqrt_recip_alphas_t = torch.sqrt(1.0 / diffusion.alphas[t]).view(-1, 1, 1)
            
            model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            
            if i == 0:
                x = model_mean
            else:
                posterior_variance_t = (1 - diffusion.alphas_cumprod[i-1]) / (1 - diffusion.alphas_cumprod[i]) * diffusion.betas[i]
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
            
        pred = x[0, -PRED_LEN:, :]

    sample_timestamps = test_dataset.timestamps[test_idx:test_idx + HISTORY_LEN + PRED_LEN]
    plot_test_sample(full_price_series[0], pred, sample_timestamps, price_scaler,
                     title="Test Sample - Improved DiffWave Forecast",
                     save_path="beschte_diffwave_forecast.png")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
