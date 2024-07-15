import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time


def load_and_preprocess_heart_data(filepath):
    try:
        data = pd.read_csv(filepath)
        X = data.drop('target', axis=1)
        y = data['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading heart data: {e}")
        return None, None, None, None


def load_and_preprocess_mobile_data(filepath):
    try:
        data = pd.read_csv(filepath)
        X = data.drop('price_range', axis=1)
        y = data['price_range']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading mobile data: {e}")
        return None, None, None, None


def run_clustering_experiment_with_timing(X_train, algorithm, n_clusters_range, n_runs=10):
    timings = []
    for n_clusters in n_clusters_range:
        total_time = 0
        for _ in range(n_runs):
            start_time = time.time()
            if algorithm == 'EM':
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            elif algorithm == 'KMeans':
                model = KMeans(n_clusters=n_clusters, random_state=42)
            else:
                raise ValueError("Unsupported algorithm: choose 'EM' or 'KMeans'")

            model.fit(X_train)
            total_time += (time.time() - start_time)

        avg_time = total_time / n_runs
        timings.append(avg_time)
    return timings

def plot_clustering_timings(timings_heart_em, timings_heart_kmeans, timings_mobile_em, timings_mobile_kmeans, n_clusters_range, save_dir):
    plt.figure(figsize=(10, 6))

    # Define color palette
    colors = sns.color_palette("husl", 2)  # Using a color palette with 2 distinct colors

    # Plotting with specific colors
    sns.lineplot(x=n_clusters_range, y=timings_heart_em, label='Heart Data - EM', color=colors[0], linestyle='-')
    sns.lineplot(x=n_clusters_range, y=timings_heart_kmeans, label='Heart Data - KMeans', color=colors[0], linestyle='--')
    sns.lineplot(x=n_clusters_range, y=timings_mobile_em, label='Mobile Data - EM', color=colors[1], linestyle='-')
    sns.lineplot(x=n_clusters_range, y=timings_mobile_kmeans, label='Mobile Data - KMeans', color=colors[1], linestyle='--')

    plt.title('Wall Clock Time')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'clustering_timings.png'))
    plt.show()



# Main execution
save_dir = '../images'
os.makedirs(save_dir, exist_ok=True)

heart_filepath = '../data/heart_statlog_cleveland_hungary_final.csv'
mobile_filepath = '../data/mobile_train.csv'

X_train_heart, X_test_heart, y_train_heart, y_test_heart = load_and_preprocess_heart_data(heart_filepath)
X_train_mobile, X_test_mobile, y_train_mobile, y_test_mobile = load_and_preprocess_mobile_data(mobile_filepath)

if X_train_heart is not None and X_train_mobile is not None:
    n_clusters_range = [2, 4, 8, 16, 32]  # Number of clusters to test

    # Run clustering experiments and measure timings
    timings_heart_em = run_clustering_experiment_with_timing(X_train_heart, 'EM', n_clusters_range)
    timings_heart_kmeans = run_clustering_experiment_with_timing(X_train_heart, 'KMeans', n_clusters_range)
    timings_mobile_em = run_clustering_experiment_with_timing(X_train_mobile, 'EM', n_clusters_range)
    timings_mobile_kmeans = run_clustering_experiment_with_timing(X_train_mobile, 'KMeans', n_clusters_range)

    # Plot the timings
    plot_clustering_timings(timings_heart_em, timings_heart_kmeans, timings_mobile_em, timings_mobile_kmeans, n_clusters_range, save_dir)
else:
    print("Data loading failed.")
