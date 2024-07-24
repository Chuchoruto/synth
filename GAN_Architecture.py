import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.parametrizations as param
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(Generator, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # LeakyReLU
            layers.append(nn.Dropout(0.3))  # Dropout layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Discriminator, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(param.spectral_norm(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i])))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # LeakyReLU
        layers.append(param.spectral_norm(nn.Linear(hidden_dims[-1], 1)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class FeatureExtractor(nn.Module):
    def __init__(self, discriminator):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(discriminator.model.children())[:-1])  # Remove the last layer

    def forward(self, x):
        return self.features(x)

def Create_GAN_Architecture(original_set):
    D = original_set.shape[1]
    generator_hidden_dim = 10 * D
    discriminator_hidden_dim = 2 * D
    num_layers = math.ceil(math.log(D)) + 1
    
    if D < 10:
        latent_dim = 200
    elif D < 18:
        latent_dim = 100
    elif D < 25:
        latent_dim = 150
    else:
        latent_dim = 200
    
    generator_hidden_dims = [math.ceil(generator_hidden_dim / (2**i)) for i in range(num_layers)]
    
    # Create discriminator hidden dimensions with one less layer and starting with 5 * D
    discriminator_hidden_dims = [math.ceil(discriminator_hidden_dim / (2**i)) for i in range(num_layers - 1)]
    
    generator = Generator(latent_dim, D, generator_hidden_dims)
    discriminator = Discriminator(D, discriminator_hidden_dims)
    
    return generator, discriminator, latent_dim


def gradient_penalty(discriminator, real_data, fake_data, lambda_gp=0.1):
    alpha = torch.rand(real_data.size(0), 1, device=real_data.device)
    alpha = alpha.expand(real_data.size())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, 
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def train_GAN_with_feature_matching(dataset, num_epochs=5000, batch_size=128, lr=0.00001, beta1=0.5, critic_updates=5, mse_weight=100, feature_weight=500, noise_std=0.1):
    dataset_np = dataset.to_numpy()
    
    generator, discriminator, latent_dim = Create_GAN_Architecture(dataset_np)
    feature_extractor = FeatureExtractor(discriminator)
    original_tensor = torch.tensor(dataset_np, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(original_tensor), batch_size=batch_size, shuffle=True)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr/2, betas=(beta1, 0.999))
    
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.999)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.999)

    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        for real_data in dataloader:
            real_data = real_data[0].to(generator.model[0].weight.device)
            current_batch_size = real_data.size(0)
            
            for _ in range(critic_updates):
                optimizer_D.zero_grad()
                z = torch.randn(current_batch_size, latent_dim).to(generator.model[0].weight.device)
                fake_data = generator(z)

                # Add noise to discriminator input
                real_data_noisy = real_data + torch.normal(0, noise_std, size=real_data.size()).to(real_data.device)
                fake_data_noisy = fake_data + torch.normal(0, noise_std, size=fake_data.size()).to(fake_data.device)
                
                # Label smoothing: real_labels = 0.9, fake_labels = 0.1
                real_labels = torch.full((current_batch_size, 1), 0.9, device=real_data.device)
                fake_labels = torch.full((current_batch_size, 1), 0.1, device=real_data.device)

                real_output = discriminator(real_data_noisy)
                fake_output = discriminator(fake_data_noisy.detach())

                real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels)
                fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels)
                gp = gradient_penalty(discriminator, real_data, fake_data, lambda_gp=0.1)
                d_loss = real_loss + fake_loss + gp
                d_loss.backward()
                optimizer_D.step()
            
            optimizer_G.zero_grad()
            fake_data = generator(z)
            gen_output = discriminator(fake_data)
            
            # Label smoothing: real_labels = 0.9
            real_labels_for_gen = torch.full((current_batch_size, 1), 0.9, device=fake_data.device)
            gen_loss = F.binary_cross_entropy_with_logits(gen_output, real_labels_for_gen)
            
            # MSE loss between generated data and a random sample from real data
            real_sample = real_data[torch.randint(0, real_data.size(0), (fake_data.size(0),))]
            mse = mse_loss(fake_data, real_sample)
            
            # Feature matching loss
            real_features = feature_extractor(real_sample)
            fake_features = feature_extractor(fake_data)
            feature_loss = mse_loss(fake_features, real_features)
            
            total_gen_loss = gen_loss + mse_weight * mse + feature_weight * feature_loss
            total_gen_loss.backward()
            optimizer_G.step()
        
        scheduler_G.step()
        scheduler_D.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {gen_loss.item():.4f} | MSE Loss: {mse.item():.4f} | Feature Loss: {feature_loss.item():.4f}")

    return generator



def train_GAN(dataset, num_epochs=5000, batch_size=128, lr=0.00001, beta1=0.5, critic_updates=5, mse_weight=10):
    dataset_np = dataset.to_numpy()
    
    generator, discriminator, latent_dim = Create_GAN_Architecture(dataset_np)
    original_tensor = torch.tensor(dataset_np, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(original_tensor), batch_size=batch_size, shuffle=True)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.99)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.99)

    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        for real_data in dataloader:
            real_data = real_data[0].to(generator.model[0].weight.device)
            current_batch_size = real_data.size(0)
            
            for _ in range(critic_updates):
                optimizer_D.zero_grad()
                z = torch.randn(current_batch_size, latent_dim).to(generator.model[0].weight.device)
                fake_data = generator(z)
                
                real_loss = -torch.mean(discriminator(real_data))
                fake_loss = torch.mean(discriminator(fake_data.detach()))
                gp = gradient_penalty(discriminator, real_data, fake_data, lambda_gp=0.1)
                d_loss = real_loss + fake_loss + gp
                d_loss.backward()
                optimizer_D.step()
            
            optimizer_G.zero_grad()
            fake_data = generator(z)
            gen_loss = -torch.mean(discriminator(fake_data))
            
            # MSE loss between generated data and a random sample from real data
            real_sample = real_data[torch.randint(0, real_data.size(0), (fake_data.size(0),))]
            mse = mse_loss(fake_data, real_sample)
            
            total_gen_loss = gen_loss + mse_weight * mse
            total_gen_loss.backward()
            optimizer_G.step()
        
        scheduler_G.step()
        scheduler_D.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {gen_loss.item():.4f} | MSE Loss: {mse.item():.4f}")

    return generator