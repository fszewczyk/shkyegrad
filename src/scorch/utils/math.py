import torch
import numpy as np

def kl_normal(mean, log_variance):
    return torch.sum(0.5 * (torch.square(mean) + torch.exp(log_variance) - log_variance - 1))

def get_normalized_data(data):
    assert data.ndim == 2
    centered_data = data - data.mean(axis=0)
    n_samples, n_features = centered_data.shape
    std_dev = np.sqrt(np.sum(centered_data ** 2) / (n_samples - 1))
    std_dev[std_dev == 0] = 1.0
    normalized_data = centered_data / std_dev
    return normalized_data
 

def get_correlation_matrix(data):
   normalized_data = get_normalized_data(data)
   return (normalized_data.T @ normalized_data) / (n_samples / 1)


def pca(data, n_components=2):
    assert data.ndim == 2
    corr_matrix = get_correlation_matrix(data)
    eig_values, eig_vectors = np.linalg.eigh(corr_matrix)

    eig_sort_idx = np.argsort(eig_values)
    eig_values = eig_values[eig_sort_idx]
    eig_vectors = eig_vectors[:, eig_sort_idx]

    normalized_data = get_normalized_data(data)
    reduced_data = normalized_data @ eig_vectors[:, :n_components]

    return eig_values, eig_vectors, reduced_data
