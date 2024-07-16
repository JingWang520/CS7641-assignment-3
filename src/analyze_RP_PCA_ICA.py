import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.size'] = 14


# Load and preprocess data
def load_and_preprocess_data(filepath, target_column):
    try:
        data = pd.read_csv(filepath)
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


# PCA analysis
def pca_analysis(X, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    return X_pca, explained_variance, pca.components_


# ICA analysis
def ica_analysis(X, n_components):
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)
    kurtosis_values = kurtosis(X_ica)
    return X_ica, kurtosis_values


# Random projection analysis
def rp_analysis(X, n_components):
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_rp = rp.fit_transform(X)
    return X_rp


def plot_pca_explained_variance(explained_variance_heart, explained_variance_mobile, save_dir):
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(explained_variance_heart) + 1), explained_variance_heart, marker='o', label='Heart Dataset')
    plt.plot(range(1, len(explained_variance_mobile) + 1), explained_variance_mobile, marker='o',
             label='Mobile Dataset')

    plt.title('PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.grid()

    filename = 'pca_explained_variance_combined.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()


def plot_ica_kurtosis(kurtosis_values_heart, kurtosis_values_mobile, save_dir):
    avg_kurtosis_values_heart = [np.mean(kurtosis_values) for kurtosis_values in kurtosis_values_heart]
    avg_kurtosis_values_mobile = [np.mean(kurtosis_values) for kurtosis_values in kurtosis_values_mobile]

    plt.figure(figsize=(10, 6))

    num_components = range(2, len(avg_kurtosis_values_heart) + 2)

    plt.plot(num_components, avg_kurtosis_values_heart, marker='o', label='Heart Dataset')
    plt.plot(num_components, avg_kurtosis_values_mobile, marker='o', label='Mobile Dataset')

    plt.title('Average ICA Kurtosis for Heart and Mobile Datasets')
    plt.xlabel('Number of Components')
    plt.ylabel('Average Kurtosis')
    plt.legend()
    plt.grid()

    filename = 'ica_avg_kurtosis_combined.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()


# Calculate reconstruction error for random projection
def calculate_reconstruction_error(X, X_rp, n_components):
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    # Fit the random projection model
    rp.fit(X)
    # Compute the pseudo-inverse of the projection matrix
    pseudo_inverse = np.linalg.pinv(rp.components_.T)
    # Reconstruct the original data
    X_rp_inverse = np.dot(X_rp, pseudo_inverse)
    # Calculate the reconstruction error
    reconstruction_error = np.mean((X - X_rp_inverse) ** 2)
    return reconstruction_error


# Plot the distribution of the first three principal components
def plot_pca_components_distribution(X_pca, dataset_name, save_dir):
    plt.figure(figsize=(14, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.histplot(X_pca[:, i], kde=True)
        plt.title(f'Distribution of PC{i + 1} for {dataset_name} Dataset')
        plt.xlabel(f'PC{i + 1}')
        plt.ylabel('Frequency')
    filename = f'pca_components_distribution_{dataset_name}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()

def plot_reconstruction_errors(errors_heart, errors_mobile, save_dir):
    plt.figure(figsize=(10, 6))

    plt.plot(range(2, 11), errors_heart, marker='o', label='Heart Dataset')
    plt.plot(range(2, 11), errors_mobile, marker='o', label='Mobile Dataset')
    plt.title('Reconstruction Error for Different n_components')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.grid()

    filename = 'reconstruction_error_combined.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()


# Main execution
save_dir = '../images'
os.makedirs(save_dir, exist_ok=True)

heart_filepath = '../data/heart_statlog_cleveland_hungary_final.csv'
mobile_filepath = '../data/mobile_train.csv'

X_heart, y_heart = load_and_preprocess_data(heart_filepath, 'target')
X_mobile, y_mobile = load_and_preprocess_data(mobile_filepath, 'price_range')

if X_heart is not None and X_mobile is not None:
    n_components = 10  # Example number of components for PCA, ICA, and RP

    # PCA analysis
    X_heart_pca, explained_variance_heart, pca_components_heart = pca_analysis(X_heart, n_components)
    X_mobile_pca, explained_variance_mobile, pca_components_mobile = pca_analysis(X_mobile, n_components)
    plot_pca_explained_variance(explained_variance_heart, explained_variance_mobile, save_dir)

    # Plot PCA components distribution
    plot_pca_components_distribution(X_heart_pca, 'Heart', save_dir)
    plot_pca_components_distribution(X_mobile_pca, 'Mobile', save_dir)

    # ICA analysis
    kurtosis_values_heart = []
    kurtosis_values_mobile = []
    for n in range(2, 11):
        _, kurtosis_heart = ica_analysis(X_heart, n)
        _, kurtosis_mobile = ica_analysis(X_mobile, n)
        kurtosis_values_heart.append(kurtosis_heart)
        kurtosis_values_mobile.append(kurtosis_mobile)
    plot_ica_kurtosis(kurtosis_values_heart, kurtosis_values_mobile, save_dir)

    # Random projection analysis and reconstruction error calculation
    reconstruction_errors_heart = []
    reconstruction_errors_mobile = []
    for n in range(2, 11):
        X_heart_rp = rp_analysis(X_heart, n)
        X_mobile_rp = rp_analysis(X_mobile, n)
        reconstruction_error_heart = calculate_reconstruction_error(X_heart, X_heart_rp, n)
        reconstruction_error_mobile = calculate_reconstruction_error(X_mobile, X_mobile_rp, n)
        reconstruction_errors_heart.append(reconstruction_error_heart)
        reconstruction_errors_mobile.append(reconstruction_error_mobile)

    plot_reconstruction_errors(reconstruction_errors_heart, reconstruction_errors_mobile, save_dir)

else:
    print("Error loading datasets.")
