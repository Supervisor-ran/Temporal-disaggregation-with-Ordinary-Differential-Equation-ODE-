import torch
import torch.nn as nn
from libs.ode_func import ODEFunction
from torchdiffeq import odeint_adjoint as odeint
# Define the VAE network with GRU encoder and ODE integration
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, freq_ratio):
        super(VAE, self).__init__()
        self.gru = nn.GRU(input_dim, 400, batch_first=True)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.ode_func = ODEFunction(latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)  # Ensure output dimension matches input_dim
        self.freq_ratio = freq_ratio

    def encode(self, x):
        h0 = torch.zeros(1, x.size(0), 400).to(x.device)  # Initialize hidden state for GRU
        out, _ = self.gru(x, h0)
        h = out[:, -1, :]  # Use the last hidden state
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)  # Ensure output dimension matches input_dim

    def forward(self, x, t):
        mu, logvar = self.encode(x)
        z0 = self.reparameterize(mu, logvar)
        # Solve ODE in latent space
        z = odeint(self.ode_func, z0, t)
        z = z[-1]  # Use the final time step
        return self.decode(z), mu, logvar

# Define the loss function with additional constraint
def loss_function(recon_x, x, mu, logvar, Y, freq_ratio):
    # Use Mean Squared Error (MSE) for continuous data
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Constraint loss
    X = recon_x.view(-1, freq_ratio)  # Reshape to [num_periods, freq_ratio]
    constraint_loss = torch.sum((torch.sum(X, dim=1) - Y) ** 2)

    return MSE + KLD + constraint_loss