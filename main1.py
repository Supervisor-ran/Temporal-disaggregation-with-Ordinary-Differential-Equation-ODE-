import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
from pre_processing import get_data, create_sequences  # 请根据实际情况调整导入路径
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from libs.utils import make_dir,normalize_tensor, denormalize_tensor_for_m, viz_check  # 请根据实际情况调整导入路径


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
    def __init__(self, latent_dim, model_noise):
        super(LatentSDE, self).__init__()
        self.ode = nn.Linear(latent_dim, latent_dim)
        self.noise_std = model_noise  # Standard deviation for noise

    def forward(self, t, z):
        noise = self.noise_std * torch.randn_like(z)
        return F.relu(self.ode(z)) + noise


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
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, rtol, atol, model_noise):
        super(GLSDE, self).__init__()
        self.encoder = EncoderGRU(input_dim, hidden_dim, latent_dim)
        self.latent_sde = LatentSDE(latent_dim, model_noise)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = mu + torch.sqrt(torch.exp(log_var))
        t = torch.linspace(0, 1, 10).to(x.device)  # Time steps for ODE solver

        z = odeint(self.latent_sde, z, t, method="dopri5", rtol=self.rtol, atol=self.atol)
        z = z[-1]  # Use the final latent state
        x_recon = self.decoder(z, x.size(1))
        return x_recon, mu, log_var


def loss_function(recon_x, x, mu, log_var, kl_need, targets=None, noise_std=0.1):
    """
    计算包含噪声处理的损失函数。

    参数:
    - recon_x: 模型重建的数据
    - x: 输入数据
    - mu: 潜在空间均值
    - log_var: 潜在空间对数方差
    - targets: 目标数据（ground truth）
    - noise_std: 噪声标准差，用于处理噪声的强度

    返回:
    - 总损失
    """

    def get_recons(recon_x, targets):

        if targets is None:
            s = (recon_x - x) ** 2
            sum_s = s.sum()
            return (1 / recon_x.shape[-2]) * sum_s
        elif targets.shape[-2] == recon_x.shape[-2]:
            recon_x_agg = torch.sum(recon_x, dim=1)
            label = targets[:, 0, :]
            print(recon_x_agg.shape)
            print(label.shape)
            assert (recon_x_agg.shape == label.shape)
            s = (recon_x_agg - label) ** 2
            sum_s = s.sum()
            return (1 / recon_x.shape[-2]) * sum_s
        else:
            print(recon_x.shape)
            print(targets.shape)
            raise Exception("Check your data, number of input data and target data should be same.")

    # 重建损失
    BCE = get_recons(recon_x, targets)

    # KL散度损失
    if kl_need == True:
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    else:
        KLD = 0

    # 噪声处理损失
    noise_loss = noise_std ** 2 * torch.sum(torch.exp(log_var))

    # 总损失
    return BCE + KLD + noise_loss


# Training function
def train(model, data_loader, optimizer, loss_function, device, kl_need, save_results=False, noise_std=0.5,
          experiment_conditions=None):
    model.train()
    total_loss = 0
    all_targets = []
    all_predictions = []

    if len(data_loader) == 2:
        for inputs in data_loader:

            inputs = inputs[0].squeeze(-1).to(device)
            print(inputs.shape)

            targets = None
            optimizer.zero_grad()
            outputs, mu, log_var = model(inputs)

            # 计算损失

            loss = loss_function(outputs, inputs, mu, log_var, targets=targets, noise_std=noise_std, kl_need=kl_need)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if save_results:
                all_predictions.append(outputs.detach().cpu().numpy())
                all_targets.append(inputs.detach().cpu().numpy())

    else:
        for inputs, targets in data_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, mu, log_var = model(inputs)

            # 计算损失

            loss = loss_function(outputs, inputs, mu, log_var, targets=targets, noise_std=noise_std, kl_need=kl_need)

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


# Function to save predictions and targets to CSV
def save_predictions_to_csv(predictions, targets, experiment_conditions=None):
    predictions = np.concatenate(predictions, axis=0)[:, 0, :]
    targets = np.concatenate(targets, axis=0)[:, 0, :]

    df = pd.DataFrame({
        'Predictions': list(predictions.reshape(-1)),
        'Targets': list(targets.reshape(-1))
    })

    df.to_csv(f'{res_path}{experiment_conditions}_predictions_vs_targets.csv', index=False)


# Function to visualize predictions and targets
def visualize_predictions(predictions, targets, experiment_conditions=None):
    predictions = np.concatenate(predictions, axis=0)[:, 0, :]
    targets = np.concatenate(targets, axis=0)[:, 0, :]

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

data, _, label_m, label_m_no_nor, label_q_nor, __, ___ = get_data(
    sequence_length)  # Ensure get_data() returns the correct label_m
inp_feat = torch.tensor(data, dtype=torch.float32)
print("input", inp_feat.shape)
label_m_tensor = torch.tensor(label_m, dtype=torch.float32).unsqueeze(-1)
label_q_tensor = torch.tensor(label_q_nor, dtype=torch.float32).unsqueeze(-1)
label_q = torch.tensor(__, dtype=torch.float32)
dataset = TensorDataset(label_q_tensor)
print(type(dataset))

# Example synthetic data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(0)
torch.manual_seed(0)


def main(dataset, experiment_conditions, input_dim, rtol, atol, model_noise, loss_noise, kl_need):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Model, optimizer and loss function
    model = GLSDE(input_dim, hidden_dim, latent_dim, output_dim, rtol, atol, model_noise).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):
        print(f"Current epoch is {epoch + 1}/{num_epochs}")
        save_results = (epoch == num_epochs - 1)
        train_loss = train(model, data_loader, optimizer, loss_function, device, kl_need=kl_need,
                           save_results=save_results, noise_std=loss_noise, experiment_conditions=experiment_conditions)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

    # Visualization of a single sample after training
    model.eval()
    # with torch.no_grad():
    #     example_data, _ = dataset[0]
    #     example_data = example_data.unsqueeze(0).to(device)
    #     reconstructed, _, _ = model(example_data)
    #     reconstructed = reconstructed.cpu().numpy().squeeze(0)
    #     original = example_data.cpu().numpy().squeeze(0)
    #
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(original, label='Original Data')
    # plt.title('Original Data')
    # plt.legend()
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(reconstructed, label='Reconstructed Data', linestyle='--')
    # plt.title('Reconstructed Data')
    # plt.legend()
    #
    # plt.savefig(f'{res_path}{experiment_conditions}_original_vs_reconstructed.png')
    # plt.show()

    return model


def get_expanded_target(model, low_freq_data):
    low_freq_data = low_freq_data.reshape(126, 1, 1)
    mu, log_var = model.encoder(low_freq_data)
    z = mu + torch.sqrt(torch.exp(log_var))

    # 生成高频数据
    t_high_freq = torch.linspace(0, 1, 10)
    z_high_freq = odeint(model.latent_sde, z, t_high_freq, method="dopri5", rtol=1e-2, atol=1e-5)

    # 使用解码器解码生成高频数据
    high_freq_data = model.decoder(z_high_freq[-1], seq_length=4)

    # 可视化原始低频数据与生成的高频数据
    plt.figure(figsize=(12, 6))

    # 可视化低频数据
    plt.subplot(2, 1, 1)
    plt.plot(low_freq_data.squeeze().numpy(), label='Low Frequency Data', color='blue')
    plt.title('Low Frequency Data (126 points)')
    plt.legend()

    # 可视化高频数据
    plt.subplot(2, 1, 2)
    plt.plot(high_freq_data.squeeze().detach().numpy(), label='High Frequency Data', linestyle='--', color='red')
    plt.title('High Frequency Data (504 points)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return high_freq_data


if __name__ == "__main__":
    expanded_target_model = main(dataset, "Expand_target_data", 1, 1e-2, 1e-5, model_noise=0.1, loss_noise=0.1,
                                 kl_need=True)
    expanded_target = get_expanded_target(expanded_target_model, label_q)
    expanded_target = normalize_tensor(expanded_target.reshape(504, 1).detach())

    expanded_target_seq = torch.tensor(create_sequences(expanded_target, 4), dtype=torch.float32)
    dataset1 = TensorDataset(inp_feat, expanded_target_seq)
    final_process = main(dataset1, "with_label_SDE", 6, 1e-1, 1e-4, model_noise=0.1, loss_noise=0.1, kl_need=False)
