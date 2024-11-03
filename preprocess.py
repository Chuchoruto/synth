import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.decomposition import PCA
import torch

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Generate embedding for a single piece of text using DistilBERT
def get_distilbert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.squeeze().numpy()

# Automatically detect columns containing text data
def detect_text_columns(df):
    text_columns = []
    for col in df.columns:
        if df[col].dtype == object:
            text_columns.append(col)
    return text_columns

# Embed and reduce text columns in the DataFrame using PCA
def embed_text_columns(df, embedding_dim=64):
    text_columns = detect_text_columns(df)
    pca = PCA(n_components=embedding_dim)
    
    for column in text_columns:
        # Generate embeddings for each text entry in the column
        embeddings = np.vstack(df[column].apply(get_distilbert_embedding).values)
        
        # Reduce the dimensionality of the embeddings
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Add reduced embeddings as individual columns
        for i in range(embedding_dim):
            df[f'{column}_embedding_{i}'] = reduced_embeddings[:, i]
        
        # Drop the original text column
        df.drop(column, axis=1, inplace=True)
    
    return df, text_columns

# Remove rows with invalid zeros in non-binary columns
def drop_invalid_zeros(df):
    binary_columns = [col for col in df.columns if set(df[col].unique()).issubset({0, 1})]
    non_binary_columns = [col for col in df.columns if col not in binary_columns]
    df_cleaned = df[~((df[non_binary_columns] == 0).any(axis=1))]
    return df_cleaned

# Remove rows with NaN values
def drop_na(df):
    return df.dropna()

# Clean dataset and embed text columns
def clean_set(df):
    df_cleaned = drop_invalid_zeros(df)
    df_cleaned = drop_na(df_cleaned)
    df_cleaned, text_columns = embed_text_columns(df_cleaned)
    return df_cleaned, text_columns

# Analyze the dataset to determine hyperparameters for GAN training
def analyze_dataset(df):
    num_samples = len(df)
    num_features = df.shape[1]
    
    # Suggest epochs based on dataset size and dimensionality
    if num_samples < 1000:
        epochs = 800
    elif num_samples < 10000:
        epochs = 2000
    else:
        epochs = 4000

    # Adjust epochs for high dimensionality
    if num_features > 50:
        epochs = int(epochs * 1.5)

    # Suggest learning rate based on dataset size and dimensionality
    lr = 0.0001
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
