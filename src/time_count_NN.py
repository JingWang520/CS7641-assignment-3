import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
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


def run_nn_experiment(X_train, y_train, X_test, y_test):
    try:
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
        start_time = time.time()
        nn.fit(X_train, y_train)
        end_time = time.time()
        y_pred = nn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        training_time = end_time - start_time
        return accuracy, cm, training_time
    except Exception as e:
        print(f"Error in NN experiment: {e}")
        return None, None, None


def run_nn_with_clustering_features(X_train, y_train, X_test, y_test, clustering_method, n_clusters):
    try:
        if clustering_method == 'EM':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        elif clustering_method == 'KMeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError("Unsupported clustering method: choose 'EM' or 'KMeans'")

        clusterer.fit(X_train)
        train_clusters = clusterer.predict(X_train).reshape(-1, 1)
        test_clusters = clusterer.predict(X_test).reshape(-1, 1)

        X_train_new = np.hstack((X_train, train_clusters))
        X_test_new = np.hstack((X_test, test_clusters))

        accuracy, cm, training_time = run_nn_experiment(X_train_new, y_train, X_test_new, y_test)
        return accuracy, cm, training_time
    except Exception as e:
        print(f"Error in NN with clustering features: {e}")
        return None, None, None


# Main execution
save_dir = '../images'
os.makedirs(save_dir, exist_ok=True)

heart_filepath = '../data/heart_statlog_cleveland_hungary_final.csv'
mobile_filepath = '../data/mobile_train.csv'

X_train_heart, X_test_heart, y_train_heart, y_test_heart = load_and_preprocess_heart_data(heart_filepath)
X_train_mobile, X_test_mobile, y_train_mobile, y_test_mobile = load_and_preprocess_mobile_data(mobile_filepath)

if X_train_heart is not None and X_train_mobile is not None:
    # Neural network with clustering features
    em_clusters = 16  # Example number of clusters
    kmean_clusters = 32  # Example number of clusters

    nn_clustering_results = []
    num_experiments = 20  # Number of times to run each experiment

    # Run experiments for original NN, EM+NN, and KMeans+NN
    for _ in range(num_experiments):
        nn_clustering_results.append(
            ('Mobile + NN', *run_nn_experiment(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile)))
        nn_clustering_results.append(('Mobile + EM Features + NN',
                                      *run_nn_with_clustering_features(X_train_mobile, y_train_mobile, X_test_mobile,
                                                                       y_test_mobile,
                                                                       'EM', em_clusters)))
        nn_clustering_results.append(('Mobile + KMeans Features + NN',
                                      *run_nn_with_clustering_features(X_train_mobile, y_train_mobile, X_test_mobile,
                                                                       y_test_mobile,
                                                                       'KMeans', kmean_clusters)))

    # Calculate average accuracy and training time
    avg_results = {}
    for result in nn_clustering_results:
        key = result[0]
        if key not in avg_results:
            avg_results[key] = {'accuracy': [], 'time': []}
        avg_results[key]['accuracy'].append(result[1])
        avg_results[key]['time'].append(result[3])

    for key in avg_results:
        avg_accuracy = np.mean(avg_results[key]['accuracy'])
        avg_time = np.mean(avg_results[key]['time'])
        print(f"{key}: Average Accuracy={avg_accuracy:.4f}, Average Training Time={avg_time:.4f} seconds")

    # Plot confusion matrices for the last run
    cm_list = [result[2] for result in nn_clustering_results[-3:]]
    titles = [result[0] for result in nn_clustering_results[-3:]]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, (cm, title) in enumerate(zip(cm_list, titles)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[i])
        axs[i].set_title(title)
        axs[i].set_xlabel('Predicted')
        axs[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
    plt.show()
else:
    pass


