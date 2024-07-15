import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, confusion_matrix
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os


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



def calculate_sse_and_log_likelihood(X, algorithm, n_clusters_range):
    sse = []
    log_likelihood = []
    for n_clusters in n_clusters_range:
        if algorithm == 'EM':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            model.fit(X)
            log_likelihood.append(model.score(X) * X.shape[0])  # Log likelihood
        elif algorithm == 'KMeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(X)
            sse.append(model.inertia_)  # Sum of squared distances to closest cluster center
        else:
            raise ValueError("Unsupported algorithm: choose 'EM' or 'KMeans'")
    return sse, log_likelihood
def plot_sse_and_log_likelihood(sse_heart_kmeans, log_likelihood_heart_em, sse_mobile_kmeans, log_likelihood_mobile_em, n_clusters_range, save_dir):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    sns.lineplot(x=n_clusters_range, y=log_likelihood_heart_em, ax=axs[0, 0])
    axs[0, 0].set_title('Heart Data - EM (Log Likelihood)')
    axs[0, 0].set_xlabel('Number of Clusters')
    axs[0, 0].set_ylabel('Log Likelihood')

    sns.lineplot(x=n_clusters_range, y=sse_heart_kmeans, ax=axs[0, 1])
    axs[0, 1].set_title('Heart Data - KMeans (SSE)')
    axs[0, 1].set_xlabel('Number of Clusters')
    axs[0, 1].set_ylabel('SSE')

    sns.lineplot(x=n_clusters_range, y=log_likelihood_mobile_em, ax=axs[1, 0])
    axs[1, 0].set_title('Mobile Data - EM (Log Likelihood)')
    axs[1, 0].set_xlabel('Number of Clusters')
    axs[1, 0].set_ylabel('Log Likelihood')

    sns.lineplot(x=n_clusters_range, y=sse_mobile_kmeans, ax=axs[1, 1])
    axs[1, 1].set_title('Mobile Data - KMeans (SSE)')
    axs[1, 1].set_xlabel('Number of Clusters')
    axs[1, 1].set_ylabel('SSE')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sse_log_likelihood_results.png'))
    plt.show()


def run_clustering_experiment(X_train, y_train, X_test, y_test, algorithm, n_clusters_range):
    results = []
    for n_clusters in n_clusters_range:
        try:
            start_time = time.time()  # Start time recording

            if algorithm == 'EM':
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            elif algorithm == 'KMeans':
                model = KMeans(n_clusters=n_clusters, random_state=42)
            else:
                raise ValueError("Unsupported algorithm: choose 'EM' or 'KMeans'")

            model.fit(X_train)
            train_labels = model.predict(X_train)
            test_labels = model.predict(X_test)

            # Map cluster labels to actual labels
            label_mapping = {}
            for cluster in np.unique(train_labels):
                mask = (train_labels == cluster)
                true_labels = y_train[mask]
                most_common_label = np.bincount(true_labels).argmax()
                label_mapping[cluster] = most_common_label

            mapped_test_labels = np.vectorize(label_mapping.get)(test_labels)
            accuracy = accuracy_score(y_test, mapped_test_labels)
            ari = adjusted_rand_score(y_test, mapped_test_labels)
            nmi = normalized_mutual_info_score(y_test, mapped_test_labels)
            fmi = fowlkes_mallows_score(y_test, mapped_test_labels)
            homogeneity = homogeneity_score(y_test, mapped_test_labels)
            completeness = completeness_score(y_test, mapped_test_labels)
            v_measure = v_measure_score(y_test, mapped_test_labels)
            silhouette = silhouette_score(X_test, test_labels) if n_clusters > 1 else None

            end_time = time.time()  # End time recording
            runtime = end_time - start_time  # Calculate runtime

            results.append((n_clusters, accuracy, ari, nmi, fmi, homogeneity, completeness, v_measure, silhouette, runtime))
        except Exception as e:
            print(f"Error in clustering experiment with {n_clusters} clusters: {e}")
            results.append((n_clusters, None, None, None, None, None, None, None, None, None))

    return results

def plot_results(results_heart_em, results_heart_kmeans, results_mobile_em, results_mobile_kmeans, save_dir):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    sns.lineplot(x=[x[0] for x in results_heart_em], y=[x[1] for x in results_heart_em], ax=axs[0, 0])
    axs[0, 0].set_title('Heart Data - EM')
    axs[0, 0].set_xlabel('Number of Clusters')
    axs[0, 0].set_ylabel('Accuracy')

    sns.lineplot(x=[x[0] for x in results_heart_kmeans], y=[x[1] for x in results_heart_kmeans], ax=axs[0, 1])
    axs[0, 1].set_title('Heart Data - KMeans')
    axs[0, 1].set_xlabel('Number of Clusters')
    axs[0, 1].set_ylabel('Accuracy')

    sns.lineplot(x=[x[0] for x in results_mobile_em], y=[x[1] for x in results_mobile_em], ax=axs[1, 0])
    axs[1, 0].set_title('Mobile Data - EM')
    axs[1, 0].set_xlabel('Number of Clusters')
    axs[1, 0].set_ylabel('Accuracy')

    sns.lineplot(x=[x[0] for x in results_mobile_kmeans], y=[x[1] for x in results_mobile_kmeans], ax=axs[1, 1])
    axs[1, 1].set_title('Mobile Data - KMeans')
    axs[1, 1].set_xlabel('Number of Clusters')
    axs[1, 1].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'clustering_results.png'))
    plt.show()



def apply_dimensionality_reduction(X_train, X_test, method, n_components=6):
    try:
        if method == 'RP':
            reducer = GaussianRandomProjection(n_components=n_components, random_state=42)
            X_train_reduced = reducer.fit_transform(X_train)
            X_test_reduced = reducer.transform(X_test)
        elif method == 'PCA':
            reducer = PCA(n_components=n_components, random_state=42)
            X_train_reduced = reducer.fit_transform(X_train)
            X_test_reduced = reducer.transform(X_test)
        elif method == 'ICA':
            reducer = FastICA(n_components=n_components, random_state=42)
            X_train_reduced = reducer.fit_transform(X_train)
            X_test_reduced = reducer.transform(X_test)
        elif method == 't-SNE':
            # Combine train and test data
            combined_data = np.vstack((X_train, X_test))
            reducer = TSNE(n_components=3, random_state=42)
            combined_reduced = reducer.fit_transform(combined_data)
            # Split the reduced data back into train and test sets
            X_train_reduced = combined_reduced[:len(X_train)]
            X_test_reduced = combined_reduced[len(X_train):]
        else:
            raise ValueError("Unsupported method: choose 'RP', 'PCA', 'ICA', or 't-SNE'")

        return X_train_reduced, X_test_reduced

    except Exception as e:
        print(f"Error in dimensionality reduction with method {method}: {e}")
        return None, None


def visualize_with_first_two_components(X, y, title, save_dir):
    try:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis')
        plt.title(title)
        plt.savefig(os.path.join(save_dir, f'{title}.png'))
        plt.show()
    except Exception as e:
        print(f"Error in visualization: {e}")



def run_clustering_with_dim_reduction(X_train, y_train, X_test, y_test, method, algorithm, n_clusters):
    if method == 'Original':
        X_train_reduced, X_test_reduced = X_train, X_test
    else:
        X_train_reduced, X_test_reduced = apply_dimensionality_reduction(X_train, X_test, method)

    if X_train_reduced is None or X_test_reduced is None:
        return None

    results = run_clustering_experiment(X_train_reduced, y_train, X_test_reduced, y_test, algorithm, [n_clusters])
    return results[0][1:]




def plot_confusion_matrices(cm_list, titles, save_dir):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, (cm, title) in enumerate(zip(cm_list, titles)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[i])
        axs[i].set_title(title)
        axs[i].set_xlabel('Predicted')
        axs[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
    plt.show()

def calculate_silhouette_scores(X, algorithm, n_clusters_range):
    silhouette_scores = []
    for n_clusters in n_clusters_range:
        if algorithm == 'EM':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        elif algorithm == 'KMeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError("Unsupported algorithm: choose 'EM' or 'KMeans'")

        model.fit(X)
        labels = model.predict(X)
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = None
        silhouette_scores.append(silhouette)
    return silhouette_scores


def plot_silhouette_scores(silhouette_heart_em, silhouette_heart_kmeans, silhouette_mobile_em, silhouette_mobile_kmeans, n_clusters_range, save_dir):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    sns.lineplot(x=n_clusters_range, y=silhouette_heart_em, ax=axs[0, 0])
    axs[0, 0].set_title('Heart Data - EM')
    axs[0, 0].set_xlabel('Number of Clusters')
    axs[0, 0].set_ylabel('Silhouette Score')

    sns.lineplot(x=n_clusters_range, y=silhouette_heart_kmeans, ax=axs[0, 1])
    axs[0, 1].set_title('Heart Data - KMeans')
    axs[0, 1].set_xlabel('Number of Clusters')
    axs[0, 1].set_ylabel('Silhouette Score')

    sns.lineplot(x=n_clusters_range, y=silhouette_mobile_em, ax=axs[1, 0])
    axs[1, 0].set_title('Mobile Data - EM')
    axs[1, 0].set_xlabel('Number of Clusters')
    axs[1, 0].set_ylabel('Silhouette Score')

    sns.lineplot(x=n_clusters_range, y=silhouette_mobile_kmeans, ax=axs[1, 1])
    axs[1, 1].set_title('Mobile Data - KMeans')
    axs[1, 1].set_xlabel('Number of Clusters')
    axs[1, 1].set_ylabel('Silhouette Score')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'silhouette_scores.png'))
    plt.show()

# Main execution
save_dir = '../images'
os.makedirs(save_dir, exist_ok=True)

heart_filepath = '../data/heart_statlog_cleveland_hungary_final.csv'
mobile_filepath = '../data/mobile_train.csv'

X_train_heart, X_test_heart, y_train_heart, y_test_heart = load_and_preprocess_heart_data(heart_filepath)
X_train_mobile, X_test_mobile, y_train_mobile, y_test_mobile = load_and_preprocess_mobile_data(mobile_filepath)

if X_train_heart is not None and X_train_mobile is not None:

    n_clusters_range = range(2, 21)  # Example range for number of clusters

    # Calculate SSE and Log Likelihood
    sse_heart_kmeans, log_likelihood_heart_em = calculate_sse_and_log_likelihood(X_train_heart, 'KMeans', n_clusters_range)
    _, log_likelihood_heart_em = calculate_sse_and_log_likelihood(X_train_heart, 'EM', n_clusters_range)
    sse_mobile_kmeans, log_likelihood_mobile_em = calculate_sse_and_log_likelihood(X_train_mobile, 'KMeans', n_clusters_range)
    _, log_likelihood_mobile_em = calculate_sse_and_log_likelihood(X_train_mobile, 'EM', n_clusters_range)

    plot_sse_and_log_likelihood(sse_heart_kmeans, log_likelihood_heart_em, sse_mobile_kmeans, log_likelihood_mobile_em, n_clusters_range, save_dir)

    # Calculate Silhouette scores
    silhouette_heart_em = calculate_silhouette_scores(X_train_heart, 'EM', n_clusters_range)
    silhouette_heart_kmeans = calculate_silhouette_scores(X_train_heart, 'KMeans', n_clusters_range)
    silhouette_mobile_em = calculate_silhouette_scores(X_train_mobile, 'EM', n_clusters_range)
    silhouette_mobile_kmeans = calculate_silhouette_scores(X_train_mobile, 'KMeans', n_clusters_range)

    plot_silhouette_scores(silhouette_heart_em, silhouette_heart_kmeans, silhouette_mobile_em, silhouette_mobile_kmeans, n_clusters_range, save_dir)

    results_heart_em = run_clustering_experiment(X_train_heart, y_train_heart, X_test_heart, y_test_heart, 'EM', n_clusters_range)
    results_heart_kmeans = run_clustering_experiment(X_train_heart, y_train_heart, X_test_heart, y_test_heart, 'KMeans', n_clusters_range)
    results_mobile_em = run_clustering_experiment(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, 'EM', n_clusters_range)
    results_mobile_kmeans = run_clustering_experiment(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, 'KMeans', n_clusters_range)

    plot_results(results_heart_em, results_heart_kmeans, results_mobile_em, results_mobile_kmeans, save_dir)

    # Dimensionality reduction and visualization
    methods = ['RP', 'PCA', 'ICA', 't-SNE', 'Original']
    for method in methods:
        if method != 'Original':
            X_train_reduced, X_test_reduced = apply_dimensionality_reduction(X_train_heart, X_test_heart, method)
            if X_train_reduced is not None:
                visualize_with_first_two_components(X_train_reduced, y_train_heart, f'Heart Data + {method}', save_dir)
            X_train_reduced, X_test_reduced = apply_dimensionality_reduction(X_train_mobile, X_test_mobile, method)
            if X_train_reduced is not None:
                visualize_with_first_two_components(X_train_reduced, y_train_mobile, f'Mobile Data + {method}', save_dir)

    # Clustering on reduced data
    em_clusters = 12  # Example number of clusters
    kmean_clusters = 12  # Example number of clusters

    results_table = []
    for method in methods:
        for algorithm in ['EM', 'KMeans']:
            if algorithm == 'EM':
                metrics = run_clustering_with_dim_reduction(X_train_heart, y_train_heart, X_test_heart, y_test_heart,
                                                            method, algorithm, em_clusters)
                if metrics is not None:
                    results_table.append((f'Heart + {method} + {algorithm}', *metrics))
                metrics = run_clustering_with_dim_reduction(X_train_mobile, y_train_mobile, X_test_mobile,
                                                            y_test_mobile,
                                                            method, algorithm, em_clusters)
                if metrics is not None:
                    results_table.append((f'Mobile + {method} + {algorithm}', *metrics))
            else:
                metrics = run_clustering_with_dim_reduction(X_train_heart, y_train_heart, X_test_heart, y_test_heart,
                                                            method, algorithm, kmean_clusters)
                if metrics is not None:
                    results_table.append((f'Heart + {method} + {algorithm}', *metrics))
                metrics = run_clustering_with_dim_reduction(X_train_mobile, y_train_mobile, X_test_mobile,
                                                            y_test_mobile, method, algorithm, kmean_clusters)
                if metrics is not None:
                    results_table.append((f'Mobile + {method} + {algorithm}', *metrics))

    # Print results table
    print("Clustering on Reduced Data Results:")
    for result in results_table:
        print(f"{result[0]}: Accuracy={result[1]:.4f},  FMI={result[4]:.4f}, "
              f"Homogeneity={result[5]:.4f}, Completeness={result[6]:.4f}, V-Measure={result[7]:.4f}, "
              f"Silhouette={result[8] if result[8] is not None else 'N/A':.4f},running time: {result[9]:.4f}")



else:
    print("Error: One or both datasets could not be loaded.")