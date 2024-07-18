import torch
from scipy.stats import ks_2samp
import pandas as pd

### Line of work
# Make all of these functions not depend on scaled data. I'll do all of the scaling and inverse within model.py

def sample_synthetic_data(generator, num_samples, latent_dim):
    z = torch.randn(num_samples, latent_dim).to(generator.model[0].weight.device)
    synthetic_data = generator(z)
    return synthetic_data.detach().cpu().numpy()

def calculate_ks(synthetic_set, original_set):
    p_values = {}
    for column in original_set.columns:
        statistic, p_value = ks_2samp(original_set[column], synthetic_set[column])
        p_values[column] = p_value
    return p_values


def selective_sample(generator, num_samples, latent_dim, original_set, max_attempts=100):
    
    
    # Generate initial synthetic data
    synthetic_data = sample_synthetic_data(generator, num_samples, latent_dim)
    synthetic_df = pd.DataFrame(synthetic_data, columns=original_set.columns)
    
    for i in range(num_samples):
        valid_sample = False
        attempts = 0
        while not valid_sample and attempts < max_attempts:
            sample = synthetic_df.iloc[i]
            valid_sample = True
            for column in original_set.columns:
                min_val = original_set[column].min()
                max_val = original_set[column].max()
                
                if sample[column] < min_val or sample[column] > max_val:
                    valid_sample = False
                    break
            
            if not valid_sample:
                # Resample the entire row
                new_sample = sample_synthetic_data(generator, 1, latent_dim)
                synthetic_df.iloc[i] = new_sample[0]
                attempts += 1
        
        if not valid_sample:
            # Handling if max attempts are reached
            print(f"Warning: Max attempts reached for sample {i}. Sample may not be valid.")

    return synthetic_df


def sample_gan(generator, num_samples, latent_dim, original_set):
    # Generate synthetic data
    synthetic_data = sample_synthetic_data(generator, num_samples, latent_dim)
    synthetic_df = pd.DataFrame(synthetic_data, columns=original_set.columns)
    
    # Check which columns are discrete
    discrete_columns = [column for column in original_set.columns if pd.api.types.is_integer_dtype(original_set[column])]
    
    # Round discrete columns to the nearest whole number
    for column in discrete_columns:
        synthetic_df[column] = synthetic_df[column].round()
    
    return synthetic_df


def round_discrete_columns(original_set, synthetic_df):
    # Check which columns are discrete
    discrete_columns = [column for column in original_set.columns if pd.api.types.is_integer_dtype(original_set[column])]

    # Round discrete columns to the nearest whole number
    for column in discrete_columns:
        synthetic_df[column] = synthetic_df[column].round()

    return synthetic_df