import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(Generator, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(True)]
        for i in range(1, len(hidden_dims)):
            layers += [nn.Linear(hidden_dims[i-1], hidden_dims[i]), nn.ReLU(True)]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Discriminator, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(True)]
        for i in range(1, len(hidden_dims)):
            layers += [nn.Linear(hidden_dims[i-1], hidden_dims[i]), nn.ReLU(True)]
        layers.append(nn.Linear(hidden_dims[-1], 1))
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
    
    hidden_dims = [math.ceil(hidden_dim / (2**i)) for i in range(num_layers)]
    #hidden_dims = [hidden_dim for i in range(num_layers)]
    
    generator = Generator(latent_dim, D, hidden_dims)
    discriminator = Discriminator(D, hidden_dims)
    
    return generator, discriminator, latent_dim

def gradient_penalty(discriminator, real_data, fake_data, lambda_gp=5):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(real_data.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(real_data.device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)

    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

def train_GAN(dataset, num_epochs=10000, batch_size=128, lr=0.0001, beta1=0.5):
    # Create the GAN architecture using the original_set
    generator, discriminator, latent_dim = Create_GAN_Architecture(dataset)
    
    # Convert the original_set to a Tensor and create a DataLoader
    original_tensor = torch.tensor(dataset.values, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(original_tensor), batch_size=batch_size, shuffle=True)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for real_data in dataloader:
            real_data = real_data[0].to(generator.model[0].weight.device)
            current_batch_size = real_data.size(0)
            
            # Train Discriminator
            for _ in range(1):  # Number of critic updates
                optimizer_D.zero_grad()
                
                z = torch.randn(current_batch_size, latent_dim).to(generator.model[0].weight.device)
                fake_data = generator(z)
                
                real_loss = -torch.mean(discriminator(real_data))
                fake_loss = torch.mean(discriminator(fake_data.detach()))
                gp = gradient_penalty(discriminator, real_data, fake_data, lambda_gp=15)
                d_loss = real_loss + fake_loss + gp
                d_loss.backward()
                
                #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # Gradient clipping
                
                optimizer_D.step()
            
            # Train Generator
            if epoch % 5 == 0:
                optimizer_G.zero_grad()
                
                gen_labels = torch.ones(current_batch_size, 1).to(generator.model[0].weight.device)  # Fool the discriminator
                gen_loss = -torch.mean(discriminator(fake_data))
                gen_loss.backward()
                optimizer_G.step()
            
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {gen_loss.item():.4f}")

    return generator

