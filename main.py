import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
from pre_processing import get_data
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
        print("GRU output shape:", h.shape)  # Debugging line
        h = h[:, -1, :]  # Use the last time step's output
        mu = self.fc1(h)
        log_var = self.fc2(h)
        return mu, log_var

# Define the Latent ODE Model with a real ODE solver
class LatentODE(nn.Module):
    def __init__(self, latent_dim):
        super(LatentODE, self).__init__()
        self.ode = nn.Linear(latent_dim, latent_dim)

    def forward(self, t, z):
        return F.relu(self.ode(z))

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, output_dim, batch_first=True)  # Output 1-dimensional data

    def forward(self, z, seq_length):
        z = F.relu(self.fc1(z))
        z = z.unsqueeze(1).repeat(1, seq_length, 1)  # Repeat for each time step
        out, _ = self.gru(z)
        return out

# Define the Generative Latent ODE model
class GLODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(GLODE, self).__init__()
        self.encoder = EncoderGRU(input_dim, hidden_dim, latent_dim)
        self.latent_ode = LatentODE(latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = mu + torch.sqrt(torch.exp(log_var))
        t = torch.linspace(0, 1, x.size(1)).to(x.device)  # Time steps for ODE solver
        z = odeint(self.latent_ode, z, t)
        z = z[-1]  # Use the final latent state
        x_recon = self.decoder(z, x.size(1))
        return x_recon, mu, log_var

def loss_function(recon_x, x, mu, log_var, targets=None):
    BCE = F.mse_loss(recon_x, targets if targets is not None else x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
# Training function
def train(model, data_loader, optimizer, loss_function, device, save_results=False):
    model.train()
    total_loss = 0

    all_targets = []
    all_predictions = []
    i = 0
    for inputs, targets in data_loader:

        inputs = inputs.to(device)
        targets = targets.squeeze(-1).to(device)
        print("Input shape:", inputs.shape)
        print("Target shape:", targets.shape)
        #这里标签也没问题

        optimizer.zero_grad()
        outputs, mu, log_var = model(inputs)
        print("output shape ", outputs.shape)

        # 计算损失
        loss = loss_function(outputs, inputs, mu, log_var, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


        # Save the true labels and predictions if save_results is True
        if save_results:
            i+=1
            all_targets.append(targets.detach().numpy())
            all_predictions.append(outputs.detach().numpy())
            print("Check all target: ", len(all_targets), "print processing count", i)
    avg_loss = total_loss / len(data_loader)


    # Save the results to CSV and visualize
    if save_results:
        save_predictions_to_csv(all_predictions, all_targets)
        visualize_predictions(all_predictions, all_targets)
    return avg_loss

# Function to save predictions and targets to CSV
def save_predictions_to_csv(predictions, targets):
    predictions = np.concatenate(predictions, axis=0)[:,0,:]
    targets = np.concatenate(targets, axis=0)[:,0,:]

    # predictions = denormalize_tensor_for_m(torch.tensor(predictions), torch.tensor(label_m_no_nor))
    # targets = denormalize_tensor_for_m(torch.tensor(targets), torch.tensor(label_m_no_nor))

    # Create a DataFrame
    df = pd.DataFrame({
        'Predictions': list(predictions.reshape(-1)),
        'Targets': list(targets.reshape(-1))
    })

    # Save to CSV
    df.to_csv(f'{res_path}{experiment_conditions}_predictions_vs_targets.csv', index=False)

# Function to visualize predictions and targets
def visualize_predictions(predictions, targets):
    predictions = np.concatenate(predictions, axis=0)[:,0,:]
    targets = np.concatenate(targets, axis=0)[:,0,:]
    print("prediction shape ", predictions.shape)
    print("target shape ", targets.shape)

    # predictions = denormalize_tensor_for_m(torch.tensor(predictions), torch.tensor(label_m_no_nor))
    # targets = denormalize_tensor_for_m(torch.tensor(targets), torch.tensor(label_m_no_nor))

    # Plot predictions vs targets
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predictions', linestyle='--', color='r')
    plt.plot(targets, label='Targets', linestyle='-', color='b')
    plt.title('Predictions vs Targets')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f'{res_path}{experiment_conditions}_predictions_vs_targets.png')
    plt.show()

# Loss function


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
experiment_conditions = "with_label_of_ground_truth"

# Example synthetic data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(0)
torch.manual_seed(0)
data, _, label_m, label_m_no_nor = get_data(sequence_length)  # Ensure get_data() returns the correct label_m
dataset = torch.tensor(data, dtype=torch.float32)  # No need to add batch dimension here
label_m_tensor = torch.tensor(label_m, dtype=torch.float32).unsqueeze(-1)  # Ensure label_m has the correct shape [504, 1]
#到这里标签数据没有问题
dataset = TensorDataset(dataset, label_m_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print(dataset, data_loader)

# Model, optimizer and loss function
model = GLODE(input_dim, hidden_dim, latent_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    print(f"Current epoch is {epoch}/{num_epochs}")
    save_results = (epoch == num_epochs - 1)  # Save results in the last epoch
    train_loss = train(model, data_loader, optimizer, loss_function, device, save_results=save_results)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

# Visualization of a single sample after training
model.eval()
with torch.no_grad():
    example_data, _ = dataset[0]
    example_data = example_data.unsqueeze(0).to(device)
    print("example_data shape ", example_data.shape)
    reconstructed, _, _ = model(example_data)
    reconstructed = reconstructed.cpu().numpy().squeeze(0)
    print("rec_data shape ", reconstructed.shape)
    original = example_data.cpu().numpy().squeeze(0)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(original, label='Original Data')
plt.title('Original Data')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(reconstructed, label='Reconstructed Data', linestyle='--')
plt.title('Reconstructed Data')
plt.legend()

# Save the plot to a file
plt.savefig(f'{res_path}{experiment_conditions}_original_vs_reconstructed.png')
plt.show()

# save_predictions_to_csv(reconstructed, )
