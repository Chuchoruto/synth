import pandas as pd
import os
from preprocess import clean_set, analyze_dataset
from GAN_Architecture import train_GAN_with_feature_matching
from sampler import calculate_ks, selective_sample, sample_gan, round_discrete_columns, decode_synthetic_data
from sklearn.preprocessing import MinMaxScaler

class Model:
    def __init__(self, csv_path, num_samples):
        self.csv_path = csv_path
        self.num_samples = num_samples
        self.initialize_model()
        self.sample_select_data()
        self.nondiscriminatory_sample()
        self.calc_ks()

    def initialize_model(self):
        # Step 1: Load and preprocess the dataset
        original = pd.read_csv(self.csv_path)
        self.original_set, self.text_columns = clean_set(original)

        # Step 2: Scale the numeric data for training
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(self.original_set)
        self.scaled_df = pd.DataFrame(scaled_data, columns=self.original_set.columns)

        # Step 3: Train the GAN model
        epochs, lr, batch_size, beta1 = analyze_dataset(self.original_set)
        self.generator = train_GAN_with_feature_matching(self.scaled_df, epochs, batch_size, lr, beta1)

    def sample_select_data(self):
        latent_dim = self.generator.model[0].in_features  # Latent dimension from the generator model
        selected_synthetic_data = selective_sample(self.generator, self.num_samples, latent_dim, self.scaled_df)
        
        # Inverse transform to original scale
        scaled_selected_data = self.scaler.inverse_transform(selected_synthetic_data)
        self.selected_synthetic_data = pd.DataFrame(scaled_selected_data, columns=self.original_set.columns)
        
        # Decode text embeddings back to original text format
        self.selected_synthetic_data = decode_synthetic_data(self.selected_synthetic_data, self.text_columns)
        
        # Round discrete columns to ensure compatibility with original data type
        self.selected_synthetic_data = round_discrete_columns(self.original_set, self.selected_synthetic_data)

    def nondiscriminatory_sample(self):
        latent_dim = self.generator.model[0].in_features
        nondisc_data = sample_gan(self.generator, self.num_samples, latent_dim, self.scaled_df)
        
        # Inverse transform to original scale
        scaled_nondisc_data = self.scaler.inverse_transform(nondisc_data)
        self.nondiscriminatory_data = pd.DataFrame(scaled_nondisc_data, columns=self.original_set.columns)
        
        # Decode text embeddings
        self.nondiscriminatory_data = decode_synthetic_data(self.nondiscriminatory_data, self.text_columns)
        
        # Round discrete columns
        self.nondiscriminatory_data = round_discrete_columns(self.original_set, self.nondiscriminatory_data)

    def calc_ks(self):
        # Calculate KS p-values to compare synthetic data with real data
        self.p_values_df = pd.DataFrame(calculate_ks(self.selected_synthetic_data, self.original_set).items(), columns=['Column', 'KS p-value'])
        self.p_values_df['KS p-value'] = self.p_values_df['KS p-value'].apply(lambda x: format(x, '.6f'))

    def get_synthetic_data(self):
        # Return the generated synthetic data
        return self.selected_synthetic_data

    def get_ks_pvalues(self):
        # Return KS test p-values for distribution comparison
        return self.p_values_df

    def check_pvalues_threshold(self):
        # Check if all p-values meet a defined threshold
        return all(float(p) < 0.05 for p in self.p_values_df['KS p-value'])
