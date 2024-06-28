# main.py

import pandas as pd
from GAN_Architecture import train_GAN, Sample_Synthetic_Data

# Placeholder path for the CSV file
csv_path = 'diabetes.csv'

# Train the GAN using the CSV file
generator = train_GAN(csv_path)

# Sample synthetic data from the trained generator
num_samples = 770  # Specify the number of samples you want to generate
latent_dim = generator.model[0].in_features  # Extract latent dimension from generator
synthetic_data = Sample_Synthetic_Data(generator, num_samples, latent_dim)

# Print the synthetic data for verification
print("Synthetic Data Samples:")
print(synthetic_data)
