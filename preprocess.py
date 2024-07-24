import pandas as pd

def drop_invalid_zeros(df):
    """
    Drops rows that have values of 0 in columns that should have a value other than 0 or 1.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with invalid rows removed.
    """
    # Identify columns with only 0s and 1s
    binary_columns = [col for col in df.columns if set(df[col].unique()).issubset({0, 1})]

    # Identify columns that should not contain 0s
    non_binary_columns = [col for col in df.columns if col not in binary_columns]

    # Drop rows with 0s in non-binary columns
    df_cleaned = df[~((df[non_binary_columns] == 0).any(axis=1))]

    return df_cleaned

def drop_na(df):
    """
    Drops rows that have NaN values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with NaN values removed.
    """
    df_cleaned = df.dropna()
    return df_cleaned

def clean_set(df):
    """
    Cleans the dataset by removing rows with invalid zeros and NaN values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = drop_invalid_zeros(df)
    df_cleaned = drop_na(df_cleaned)
    return df_cleaned



def analyze_dataset(df):
    """
    Analyzes the dataset to suggest hyperparameters for GAN training.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    dict: Suggested hyperparameters.
    """
    num_samples = len(df)
    num_features = df.shape[1]
    
    # Suggest epochs based on dataset size and dimensionality
    if num_samples < 1000:
        epochs = 1000
    elif num_samples < 10000:
        epochs = 5000
    else:
        epochs = 10000

    # Adjust epochs for high dimensionality
    if num_features > 50:
        epochs = int(epochs * 1.5)

    # Suggest learning rate based on dataset size and dimensionality
    if num_samples < 1000:
        lr = 0.0001
    elif num_samples < 10000:
        lr = 0.00001
    else:
        lr = 0.000001

    # Adjust learning rate for high dimensionality
    if num_features > 50:
        lr = lr / 2

    # Suggest batch size based on dataset size and dimensionality
    if num_samples < 1000:
        batch_size = 128
    elif num_samples < 10000:
        batch_size = 512
    else:
        batch_size = 1048

    # Adjust batch size for high dimensionality
    if num_features > 50:
        batch_size = min(64, batch_size)

    # Beta1 value for Adam optimizer
    beta1 = 0.5

    return epochs, lr, batch_size, beta1