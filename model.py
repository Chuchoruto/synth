import pandas as pd
import os
from preprocess import clean_set, analyze_dataset
from GAN_Architecture import train_GAN, train_GAN_with_feature_matching
from sampler import calculate_ks, selective_sample, sample_gan
from sklearn.preprocessing import MinMaxScaler

class Model:
    def __init__(self, csv_path, num_samples):
        self.csv_path = csv_path
        self.run_training(csv_path, 2)
        self.sample_select_data(num_samples)
        self.nondiscriminatory_sample(num_samples)
        self.calc_ks()

    def run_training(self, csv_path, type):
        original = pd.read_csv(csv_path)
        self.original_set = clean_set(original)

        # Step 1: Scale the original dataset
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(self.original_set)
        self.scaled_df = pd.DataFrame(scaled_data, columns=self.original_set.columns)

        epochs, lr, batch_size, beta1 = analyze_dataset(self.original_set)
        if type == 1:
            self.generator = train_GAN(self.scaled_df, epochs, batch_size, lr, beta1)
        else:
            self.generator = train_GAN_with_feature_matching(self.scaled_df, epochs, batch_size, lr, beta1)

    def nondiscriminatory_sample(self, num_samples):
        latent_dim = self.generator.model[0].in_features  # Extract latent dimension from generator

        nondisc_data = sample_gan(self.generator, num_samples, latent_dim, self.scaled_df)
        scaled_nondisc_data = self.scaler.inverse_transform(nondisc_data)
        self.nondiscriminatory_data = pd.DataFrame(scaled_nondisc_data, columns=self.original_set.columns)

    def sample_select_data(self, num_samples):
        latent_dim = self.generator.model[0].in_features  # Extract latent dimension from generator

        selected_synthetic_data = selective_sample(self.generator, num_samples, latent_dim, self.scaled_df)
        scaled_selected_data = self.scaler.inverse_transform(selected_synthetic_data)
        self.selected_synthetic_data = pd.DataFrame(scaled_selected_data, columns=self.original_set.columns)

    def calc_ks(self):
        
        p_values = calculate_ks(self.selected_synthetic_data, self.original_set)
        # Convert the p-values dictionary to a DataFrame for columnar representation
        self.p_values_df = pd.DataFrame(list(p_values.items()), columns=['Column', 'KS p-value'])
        # Round the p-values to 4 decimal places and convert to string to avoid scientific notation
        self.p_values_df['KS p-value'] = self.p_values_df['KS p-value'].apply(lambda x: format(x, '.6f'))

    def get_synthetic_csv(self):
        # Extract the original filename
        original_filename = os.path.basename(self.csv_path)
        # Create the new filename
        new_filename = f"synthetic_{original_filename}"
        # Save the synthetic data to a CSV file
        self.synthetic_csv = self.selected_synthetic_data.to_csv(new_filename, index=False)
        # Return the new filename
        return self.synthetic_csv

    def get_ks_pvalues(self):
        return self.p_values_df

    def check_pvalues_threshold(self):
        return all(float(p) < 0.05 for p in self.p_values_df['KS p-value'])

