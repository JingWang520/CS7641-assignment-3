import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
from sklearn.manifold import TSNE


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
def apply_dimensionality_reduction(X_train, X_test, method, n_components=6):
    try:
        if method == 'RP':
            reducer = GaussianRandomProjection(n_components=n_components, random_state=42)
        elif method == 'PCA':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'ICA':
            reducer = FastICA(n_components=n_components, random_state=42)
        elif method == 'tSNE':
            reducer = TSNE(n_components=n_components, random_state=42, n_iter=300)
        else:
            raise ValueError("Unsupported method: choose 'RP', 'PCA', 'ICA', or 'tSNE'")

        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test) if method != 'tSNE' else reducer.fit_transform(X_test)
        return X_train_reduced, X_test_reduced
    except Exception as e:
        print(f"Error in dimensionality reduction with method {method}: {e}")
        return None, None

def measure_reduction_time(X_train, X_test, method, n_components_range, n_runs=30):
    times = []
    for n_components in n_components_range:
        total_time = 0
        for _ in range(n_runs):
            start_time = time.time()
            apply_dimensionality_reduction(X_train, X_test, method, n_components)
            total_time += time.time() - start_time
        avg_time = total_time / n_runs
        times.append(avg_time)
    while len(times) < 9:
        times.append(None)

    return times

def plot_reduction_times(times_dict, n_components_range, title, save_dir):
    plt.figure(figsize=(10, 6))
    for method, times in times_dict.items():
        sns.lineplot(x=n_components_range, y=times, label=method)
    plt.title(title)
    plt.xlabel('Number of Components')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    plt.show()
# Main execution
save_dir = '../images'
os.makedirs(save_dir, exist_ok=True)

heart_filepath = '../data/heart_statlog_cleveland_hungary_final.csv'
mobile_filepath = '../data/mobile_train.csv'

X_train_heart, X_test_heart, y_train_heart, y_test_heart = load_and_preprocess_heart_data(heart_filepath)
X_train_mobile, X_test_mobile, y_train_mobile, y_test_mobile = load_and_preprocess_mobile_data(mobile_filepath)

if X_train_heart is not None and X_train_mobile is not None:
    methods = ['RP', 'PCA', 'ICA', 'tSNE']
    n_components_range = [2,3, 4, 5, 6, 7, 8, 9, 10]

    # Measure reduction times for heart data
    times_heart = {}
    for method in methods:
        if method == 'tSNE':
            times_heart[method] = measure_reduction_time(X_train_heart, X_test_heart, method, [2, 3], n_runs=5)
        else:
            times_heart[method] = measure_reduction_time(X_train_heart, X_test_heart, method, n_components_range)

    # Measure reduction times for mobile data
    times_mobile = {}
    for method in methods:
        if method == 'tSNE':
            times_mobile[method] = measure_reduction_time(X_train_mobile, X_test_mobile, method, [2, 3], n_runs=5)
        else:
            times_mobile[method] = measure_reduction_time(X_train_mobile, X_test_mobile, method, n_components_range)

    # Plot reduction times for heart data
    plot_reduction_times(times_heart, n_components_range, 'Dimensionality Reduction Wall Clock Times (Heart Data)', save_dir)

    # Plot reduction times for mobile data
    plot_reduction_times(times_mobile, n_components_range, 'Dimensionality Reduction Wall Clock Times (Mobile Data)', save_dir)
else:
    pass