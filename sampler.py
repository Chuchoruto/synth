import torch
from scipy.stats import ks_2samp
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize decoder model and tokenizer for text embedding decoding
decoder_tokenizer = AutoTokenizer.from_pretrained("t5-small")
decoder_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Decode embeddings to text
def decode_embeddings(embeddings):
    decoded_texts = []
    for embedding in embeddings:
        inputs = decoder_tokenizer(" ".join(map(str, embedding)), return_tensors="pt")
        outputs = decoder_model.generate(inputs.input_ids, max_length=50)
        decoded_texts.append(decoder_tokenizer.decode(outputs[0], skip_special_tokens=True))
    return decoded_texts

# Decode synthetic data embeddings back to original text for specific columns
def decode_synthetic_data(synthetic_data, text_columns, embedding_dim=64):
    for column in text_columns:
        # Retrieve embedding columns for the text field
        embedding_cols = [f"{column}_embedding_{i}" for i in range(embedding_dim)]
        embeddings = synthetic_data[embedding_cols].values
        decoded_texts = decode_embeddings(embeddings)
        synthetic_data[column] = decoded_texts
        synthetic_data.drop(embedding_cols, axis=1, inplace=True)
    return synthetic_data

# Sample synthetic data from the generator
def sample_synthetic_data(generator, num_samples, latent_dim):
    z = torch.randn(num_samples, latent_dim).to(generator.model[0].weight.device)
    synthetic_data = generator(z)
    return synthetic_data.detach().cpu().numpy()

# Kolmogorov-Smirnov test to compare distributions between real and synthetic data
def calculate_ks(synthetic_set, original_set):
    p_values = {}
    for column in original_set.columns:
        statistic, p_value = ks_2samp(original_set[column], synthetic_set[column])
        p_values[column] = p_value
    return p_values

# Generate a selectively sampled synthetic dataset
def selective_sample(generator, num_samples, latent_dim, original_set, max_attempts=100):
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
            print(f"Warning: Max attempts reached for sample {i}. Sample may not be valid.")

    return synthetic_df

# Generate synthetic data and round discrete columns
def sample_gan(generator, num_samples, latent_dim, original_set):
    synthetic_data = sample_synthetic_data(generator, num_samples, latent_dim)
    synthetic_df = pd.DataFrame(synthetic_data, columns=original_set.columns)
    
    discrete_columns = [column for column in original_set.columns if pd.api.types.is_integer_dtype(original_set[column])]
    
    for column in discrete_columns:
        synthetic_df[column] = synthetic_df[column].round()
    
    return synthetic_df

# Round discrete columns to nearest integer for consistency
def round_discrete_columns(original_set, synthetic_df):
    discrete_columns = [column for column in original_set.columns if pd.api.types.is_integer_dtype(original_set[column])]

    for column in discrete_columns:
        synthetic_df[column] = synthetic_df[column].round()

    return synthetic_df
