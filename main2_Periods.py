import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
from pre_processing import get_data, create_sequences
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from libs.utils import make_dir, denormalize_tensor_for_m, viz_check

# Define the Encoder with GRU
class EncoderGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        h = h[:, -1, :]  # Use the last time step's output
        mu = self.fc1(h)
        log_var = self.fc2(h)
        return mu, log_var

# Define the Latent SDE Model with a real ODE solver
class LatentSDE(nn.Module):
    def __init__(self, latent_dim, trend_dim, trend):
        super(LatentSDE, self).__init__()
        self.ode = nn.Linear(latent_dim + trend_dim, latent_dim)
        self.noise_std = 0.1
        self.period = 1.0
        self.trend = trend

    def forward(self, t, z):
        z, trend = z.chunk(2, dim=-1)
        noise = self.noise_std * torch.randn_like(z)
        periodic_component = torch.sin(2 * torch.pi * t / self.period)
        z_with_trend = torch.cat([z, self.trend], dim=-1)
        return F.relu(self.ode(z_with_trend)) + noise + periodic_component

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, output_dim, batch_first=True)

    def forward(self, z, seq_length):
        z = F.relu(self.fc1(z))
        z = z.unsqueeze(1).repeat(1, seq_length, 1)  # Repeat for each time step
        out, _ = self.gru(z)
        return out

# Define the Generative Latent SDE model
class GLSDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, rtol, atol, trend_dim):
        super(GLSDE, self).__init__()
        self.encoder = EncoderGRU(input_dim, hidden_dim, latent_dim)
        self.latent_sde = LatentSDE(latent_dim, trend_dim, trend)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.rtol = rtol
        self.atol = atol


    def forward(self, x, trend):
        mu, log_var = self.encoder(x)
        z = mu + torch.sqrt(torch.exp(log_var))
        t = torch.linspace(0, 1, 10).to(x.device)
        trend_tensor = trend.unsqueeze(0).repeat(t.shape[0], 1, 1).to(x.device)
        print("trend tensor", trend_tensor.shape)
        print("z", z.shape)
        z_with_trend = torch.cat([z, trend_tensor], dim=-1)
        z = odeint(self.latent_sde, z_with_trend, t, method="dopri5", rtol=self.rtol, atol=self.atol)
        z = z[-1][:, :z.size(-1) // 2]  # Extract z (the first half)
        x_recon = self.decoder(z, x.size(1))
        return x_recon, mu, log_var

def loss_function(recon_x, x, mu, log_var, targets=None, noise_std=0.1):
    def get_recons(recon_x, targets):
        if targets is None:
            s = (recon_x - x)**2
            return (1/recon_x.shape[-2]) * s.sum()
        elif targets.shape[-2] == recon_x.shape[-2]:
            recon_x_agg = torch.sum(recon_x, dim=1)
            label = targets[:,0,:]
            s = (recon_x_agg - label) ** 2
            return (1 / recon_x.shape[-2]) * s.sum()
        else:
            raise ValueError("Check your data, number of input data and target data should be same.")

    BCE = get_recons(recon_x, targets)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    noise_loss = noise_std ** 2 * torch.sum(torch.exp(log_var))
    return BCE + KLD + noise_loss

def train(model, data_loader, optimizer, loss_function, device, save_results=False, noise_std=0.1, experiment_conditions=None):
    model.train()
    total_loss = 0
    all_targets = []
    all_predictions = []

    for batch in data_loader:
        if len(batch) == 2:
            inputs, targets = batch
            targets = targets.to(device)
        else:
            inputs = batch[0].squeeze(-1).to(device)
            targets = None

        inputs = inputs.to(device)
        optimizer.zero_grad()  # Ensure gradients are zeroed before backward pass
        outputs, mu, log_var = model(inputs)
        loss = loss_function(outputs, inputs, mu, log_var, targets, noise_std=noise_std)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if save_results:
            all_targets.append(targets.detach().cpu().numpy())
            all_predictions.append(outputs.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    if save_results:
        save_predictions_to_csv(all_predictions, all_targets, experiment_conditions)
        visualize_predictions(all_predictions, all_targets, experiment_conditions)

    return avg_loss

def save_predictions_to_csv(predictions, targets, experiment_conditions=None):
    predictions = np.concatenate(predictions, axis=0)[:,0,:]
    targets = np.concatenate(targets, axis=0)[:,0,:]

    df = pd.DataFrame({
        'Predictions': list(predictions.reshape(-1)),
        'Targets': list(targets.reshape(-1))
    })

    df.to_csv(f'{res_path}{experiment_conditions}_predictions_vs_targets.csv', index=False)

def visualize_predictions(predictions, targets, experiment_conditions=None):
    predictions = np.concatenate(predictions, axis=0)[:,0,:]
    targets = np.concatenate(targets, axis=0)[:,0,:]

    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predictions', linestyle='--', color='r')
    plt.plot(targets, label='Targets', linestyle='-', color='b')
    plt.title('Predictions vs Targets')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{res_path}{experiment_conditions}_predictions_vs_targets.png')
    plt.show()

# Hyperparameters
input_dim = 6
hidden_dim = 64
latent_dim = 16
output_dim = 1
learning_rate = 1e-3
num_epochs = 1000
batch_size = 64
sequence_length = 4
res_path = "generated_res/"
make_dir(res_path)

data, _, label_m, label_m_no_nor, label_q_nor, __, ___ = get_data(sequence_length)
inp_feat = torch.tensor(data, dtype=torch.float32)
label_m_tensor = torch.tensor(label_m, dtype=torch.float32).unsqueeze(-1)
label_q_tensor = torch.tensor(label_q_nor, dtype=torch.float32).unsqueeze(-1)
label_q = torch.tensor(__, dtype=torch.float32).clone().detach()
dataset = TensorDataset(label_q_tensor)

# Example synthetic data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(0)
torch.manual_seed(0)

def main(dataset, experiment_conditions, input_dim, rtol, atol, trend_dim, trend):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = GLSDE(input_dim, hidden_dim, latent_dim, output_dim, rtol, atol, trend_dim, trend).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Current epoch is {epoch + 1}/{num_epochs}")
        save_results = (epoch == num_epochs - 1)
        train_loss = train(model, data_loader, optimizer, loss_function, device, save_results=save_results, noise_std=0.1, experiment_conditions=experiment_conditions)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

    model.eval()
    return model

def get_expanded_target(model, low_freq_data, trend):
    low_freq_data = low_freq_data.reshape(126, 1, 1)
    mu, log_var = model.encoder(low_freq_data)
    z = mu + torch.sqrt(torch.exp(log_var))

    t_high_freq = torch.linspace(0, 1, 10).to(low_freq_data.device)
    trend_tensor = torch.tensor(trend).unsqueeze(0).repeat(t_high_freq.shape[0], 1, 1).to(low_freq_data.device)
    z_high_freq = odeint(model.latent_sde, z, t_high_freq, method="dopri5", rtol=1e-2, atol=1e-5, args=(trend_tensor,))

    high_freq_data = model.decoder(z_high_freq[-1], seq_length=4)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(low_freq_data.squeeze().numpy(), label='Low Frequency Data', color='blue')
    plt.title('Low Frequency Data (126 points)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(high_freq_data.squeeze().detach().numpy(), label='High Frequency Data', linestyle='--', color='red')
    plt.title('High Frequency Data (504 points)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return high_freq_data

from statsmodels.tsa.seasonal import STL

def stl_decompose(data, period):
    stl = STL(data, period=period)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid

if __name__ == "__main__":
    trend_q, seas_q, resid_q = stl_decompose(__, period=4)
    print(trend_q.shape)

    expanded_target_model = main(dataset, "Expand_target_data", 1, 1e-2, 1e-5, trend_dim=1, trend=torch.tensor(trend_q, dtype=torch.float32))
    expanded_target = get_expanded_target(expanded_target_model, label_q, trend_q)
    expanded_target = expanded_target.reshape(504, 1).detach().numpy()
    trend, seas, resid = stl_decompose(expanded_target.squeeze(), period=4)
    expanded_target_seq = torch.tensor(create_sequences(expanded_target, 4), dtype=torch.float32)
    dataset1 = TensorDataset(inp_feat, expanded_target_seq)
    final_process = main(dataset1, "with_label_SDE", 6, 1e-2, 1e-5, trend_dim=1, trend=torch.tensor(trend, dtype=torch.float32))
