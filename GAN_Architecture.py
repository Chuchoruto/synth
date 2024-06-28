# GAN_Architecture.py

import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(True)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(True)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def Create_GAN_Architecture(original_set):
    D = original_set.shape[1]
    hidden_dim = 7 * D
    num_layers = math.ceil(math.log(D))
    
    if D < 10:
        latent_dim = 50
    elif D < 18:
        latent_dim = 100
    elif D < 25:
        latent_dim = 150
    else:
        latent_dim = 200
    
    generator = Generator(latent_dim, D, hidden_dim, num_layers)
    discriminator = Discriminator(D, hidden_dim, num_layers)
    
    return generator, discriminator, latent_dim

def train_GAN(csv_path, num_epochs=10000, batch_size=128, lr=0.0001, beta1=0.5):
    # Read the CSV file into a pandas DataFrame
    original_set = pd.read_csv(csv_path)

    # Create the GAN architecture using the original_set
    generator, discriminator, latent_dim = Create_GAN_Architecture(original_set)
    
    # Convert the original_set to a Tensor and create a DataLoader
    original_tensor = torch.tensor(original_set.values, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(original_tensor), batch_size=batch_size, shuffle=True)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for real_data in dataloader:
            real_data = real_data[0]
            # Train Discriminator
            optimizer_D.zero_grad()
            
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            real_loss = criterion(discriminator(real_data), real_labels)
            fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            gen_labels = torch.ones(batch_size, 1)  # Fool the discriminator
            gen_loss = criterion(discriminator(fake_data), gen_labels)
            gen_loss.backward()
            optimizer_G.step()
            
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {gen_loss.item():.4f}")

    return generator

def Sample_Synthetic_Data(generator, num_samples, latent_dim):
    z = torch.randn(num_samples, latent_dim)
    synthetic_data = generator(z)
    return synthetic_data.detach().numpy()


