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