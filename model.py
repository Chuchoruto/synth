import pandas as pd
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
        self.p_values_df['KS p-value'] = self.p_values_df['KS p-value'].apply(lambda x: format(x, '.4f'))

