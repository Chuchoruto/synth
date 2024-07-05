# main.py

import pandas as pd
from GAN_Architecture import train_GAN, Sample_Synthetic_Data, CalculateKS

# Placeholder path for the CSV file
csv_path = 'diabetes.csv'

# Train the GAN using the CSV file
generator = train_GAN(csv_path)

# Sample synthetic data from the trained generator
num_samples = 770  # Specify the number of samples you want to generate
latent_dim = generator.model[0].in_features  # Extract latent dimension from generator
synthetic_data = Sample_Synthetic_Data(generator, num_samples, latent_dim)

# Load the original dataset for comparison
original_set = pd.read_csv(csv_path)

# Convert the synthetic data to a DataFrame with the same column names as the original dataset
synthetic_set = pd.DataFrame(synthetic_data, columns=original_set.columns)

# Calculate KS test p-values
p_values = CalculateKS(original_set, synthetic_set)

# Print the p-values for verification
print("Kolmogorov-Smirnov Test p-values:")
print(p_values)
