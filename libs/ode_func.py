import torch.nn as nn
# Define the ODE function
class ODEFunction(nn.Module):
    def __init__(self, latent_dim):
        super(ODEFunction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
            )

    def forward(self, t, z):
        return self.fc(z)