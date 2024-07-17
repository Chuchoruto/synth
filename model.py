import pandas as pd
import os
from preprocess import clean_set, analyze_dataset
from GAN_Architecture import train_GAN, CalculateKS, selective_sample


class Model:
    def __init__(self, csv_path, num_samples):
        self.run_training(csv_path)
        self.sample_select_data(num_samples)
        self.calc_KS()



    def run_training(self, csv_path):
        original = pd.read_csv(csv_path)

        self.original_set = clean_set(original)

        epochs, lr, batch_size, beta1 = analyze_dataset(self.original_set)

        self.generator = train_GAN(self.original_set, epochs, batch_size, lr, beta1)



    def sample_select_data(self, num_samples):
        latent_dim = self.generator.model[0].in_features  # Extract latent dimension from generator

        selected_synthetic_data = selective_sample(self.generator, num_samples, latent_dim, self.original_set)
        self.selected_synthetic_data = pd.DataFrame(selected_synthetic_data, columns=self.original_set.columns)

    
    def calc_KS(self):
        p_values = CalculateKS(self.original_set, self.selected_synthetic_data)
        # Convert the p-values dictionary to a DataFrame for columnar representation
        self.p_values_df = pd.DataFrame(list(p_values.items()), columns=['Column', 'KS p-value'])
        # Round the p-values to 4 decimal places and convert to string to avoid scientific notation
        self.p_values_df['KS p-value'] = self.p_values_df['KS p-value'].apply(lambda x: format(x, '.6f'))


    ## FUNCTION that creates a CSV of the synthetic data
    def get_synthetic_csv(self):
        # Extract the original filename
        original_filename = os.path.basename(self.csv_path)
        # Create the new filename
        new_filename = f"synthetic_{original_filename}"
        # Save the synthetic data to a CSV file
        self.synthetic_csv = self.selected_synthetic_data.to_csv(new_filename, index=False)
        # Return the new filename
        return self.synthetic_csv
    

    def get_KS_pvalues(self):
        return self.p_values_df
    
    def check_pvalues_threshold(self):
        return all(float(p) < 0.05 for p in self.p_values_df['KS p-value'])
