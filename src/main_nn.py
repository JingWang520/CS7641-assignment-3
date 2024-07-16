import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import os
import psutil


plt.rcParams['font.size'] = 16
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


def run_nn_experiment(X_train, y_train, X_test, y_test):
    try:
        if X_train is None or X_test is None:
            raise ValueError("Training or testing data is None.")

        nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, cm
    except Exception as e:
        print(f"Error in NN experiment: {e}")
        return None, None


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

        accuracy, cm = run_nn_experiment(X_train_new, y_train, X_test_new, y_test)
        return accuracy, cm
    except Exception as e:
        print(f"Error in NN with clustering features: {e}")
        return None, None

def run_nn_experiment_with_components(X_train, y_train, X_test, y_test, method, n_components_range):
    results = []
    times = []
    for n_components in n_components_range:
        start_time = time.time()
        X_train_reduced, X_test_reduced = apply_dimensionality_reduction(X_train, X_test, method, n_components)
        if X_train_reduced is not None:
            accuracy, _ = run_nn_experiment(X_train_reduced, y_train, X_test_reduced, y_test)
            if accuracy is not None:
                results.append((n_components, accuracy))
                elapsed_time = time.time() - start_time
                times.append((n_components, elapsed_time))
    return results, times


def plot_nn_results(nn_results, title, save_dir):
    plt.figure(figsize=(10, 6))
    for method, results in nn_results.items():
        n_components = [x[0] for x in results]
        accuracies = [x[1] for x in results]
        sns.lineplot(x=n_components, y=accuracies, label=method)
    plt.title(title)
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    plt.show()


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


def plot_loss_curves(X_train, y_train, X_test, y_test, methods, n_components_dict, save_dir):
    plt.figure(figsize=(10, 6))

    # Baseline without dimensionality reduction
    train_sizes = np.linspace(0.1, 0.99, 10)
    baseline_losses = []
    for train_size in train_sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size,
                                                                random_state=42)
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
        nn.fit(X_train_subset, y_train_subset)
        y_pred_proba = nn.predict_proba(X_test)
        loss = log_loss(y_test, y_pred_proba)
        baseline_losses.append(loss)
    sns.lineplot(x=train_sizes, y=baseline_losses, label='Baseline NN')

    for method, n_components in n_components_dict.items():
        losses = []
        for train_size in train_sizes:
            X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size,
                                                                    random_state=42)
            X_train_reduced, X_test_reduced = apply_dimensionality_reduction(X_train_subset, X_test, method,
                                                                             n_components)
            nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
            nn.fit(X_train_reduced, y_train_subset)
            y_pred_proba = nn.predict_proba(X_test_reduced)
            loss = log_loss(y_test, y_pred_proba)
            losses.append(loss)
        sns.lineplot(x=train_sizes, y=losses, label=method)

    plt.title('Loss Curves (In Testing Dataset)')
    plt.xlabel('Training Set Size')
    plt.ylabel('Testing Loss Value')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.show()


def plot_learning_curves(X_train, y_train, X_test, y_test, methods, n_components_dict, save_dir):
    plt.figure(figsize=(10, 6))

    # Baseline without dimensionality reduction
    train_sizes = np.linspace(0.1, 0.99, 10)
    baseline_train_accuracies = []
    baseline_test_accuracies = []
    for train_size in train_sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size,
                                                                random_state=42)
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
        nn.fit(X_train_subset, y_train_subset)
        baseline_train_accuracies.append(nn.score(X_train_subset, y_train_subset))
        baseline_test_accuracies.append(nn.score(X_test, y_test))
    sns.lineplot(x=train_sizes, y=baseline_train_accuracies, label='Baseline NN Train')
    sns.lineplot(x=train_sizes, y=baseline_test_accuracies, label='Baseline NN Test')

    for method, n_components in n_components_dict.items():
        train_accuracies = []
        test_accuracies = []
        for train_size in train_sizes:
            X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size,
                                                                    random_state=42)
            X_train_reduced, X_test_reduced = apply_dimensionality_reduction(X_train_subset, X_test, method,
                                                                             n_components)
            nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
            nn.fit(X_train_reduced, y_train_subset)
            train_accuracies.append(nn.score(X_train_reduced, y_train_subset))
            test_accuracies.append(nn.score(X_test_reduced, y_test))
        sns.lineplot(x=train_sizes, y=train_accuracies, label=f'{method} Train')
        sns.lineplot(x=train_sizes, y=test_accuracies, label=f'{method} Test')

    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.show()
def plot_loss_curves_with_clustering(X_train, y_train, X_test, y_test, clustering_methods, n_clusters_dict, save_dir):
    plt.figure(figsize=(10, 6))
    for method, n_clusters in n_clusters_dict.items():
        train_sizes = np.linspace(0.1, 0.99, 10)
        losses = []
        for train_size in train_sizes:
            X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
            if method == 'Mobile + NN':
                X_train_new, X_test_new = X_train_subset, X_test
            else:
                if 'EM' in method:
                    clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                elif 'KMeans' in method:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                clusterer.fit(X_train_subset)
                train_clusters = clusterer.predict(X_train_subset).reshape(-1, 1)
                test_clusters = clusterer.predict(X_test).reshape(-1, 1)
                X_train_new = np.hstack((X_train_subset, train_clusters))
                X_test_new = np.hstack((X_test, test_clusters))
            nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
            nn.fit(X_train_new, y_train_subset)
            y_pred_proba = nn.predict_proba(X_test_new)
            loss = log_loss(y_test, y_pred_proba)
            losses.append(loss)
        sns.lineplot(x=train_sizes, y=losses, label=method)
    plt.title('Loss Curves (In Testing Dataset) with Clustering Features')
    plt.xlabel('Training Set Size')
    plt.ylabel('Testing Loss Value')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves_with_clustering.png'))
    plt.show()


def plot_learning_curves_with_clustering(X_train, y_train, X_test, y_test, clustering_methods, n_clusters_dict, save_dir):
    plt.figure(figsize=(10, 6))
    for method, n_clusters in n_clusters_dict.items():
        train_sizes = np.linspace(0.1, 0.99, 10)
        train_accuracies = []
        test_accuracies = []
        for train_size in train_sizes:
            X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
            if method == 'Mobile + NN':
                X_train_new, X_test_new = X_train_subset, X_test
            else:
                if 'EM' in method:
                    clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                elif 'KMeans' in method:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                clusterer.fit(X_train_subset)
                train_clusters = clusterer.predict(X_train_subset).reshape(-1, 1)
                test_clusters = clusterer.predict(X_test).reshape(-1, 1)
                X_train_new = np.hstack((X_train_subset, train_clusters))
                X_test_new = np.hstack((X_test, test_clusters))
            nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=300, random_state=42)
            nn.fit(X_train_new, y_train_subset)
            train_accuracies.append(nn.score(X_train_new, y_train_subset))
            test_accuracies.append(nn.score(X_test_new, y_test))
        sns.lineplot(x=train_sizes, y=train_accuracies, label=f'{method} Train')
        sns.lineplot(x=train_sizes, y=test_accuracies, label=f'{method} Test')
    plt.title('Learning Curves with Clustering Features')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'learning_curves_with_clustering.png'))
    plt.show()


def plot_memory_usage_with_clustering(clustering_methods, n_clusters_dict, X_train, X_test, save_dir):
    memory_usage = []
    process = psutil.Process(os.getpid())

    for method, n_clusters in n_clusters_dict.items():
        # Get memory usage before applying clustering
        mem_before = process.memory_info().rss

        if method == 'Mobile + NN':
            _ = X_train, X_test
        else:
            if 'EM' in method:
                clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            elif 'KMeans' in method:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)

            clusterer.fit(X_train)
            train_clusters = clusterer.predict(X_train).reshape(-1, 1)
            test_clusters = clusterer.predict(X_test).reshape(-1, 1)
            X_train_new = np.hstack((X_train, train_clusters))
            X_test_new = np.hstack((X_test, test_clusters))

        # Get memory usage after applying clustering
        mem_after = process.memory_info().rss

        # Calculate the absolute increase in memory usage
        mem_increase = abs(mem_after - mem_before) / 1024 ** 2  # Convert to MB
        memory_usage.append(mem_increase)

    # Plotting the memory usage
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(n_clusters_dict.keys()), y=memory_usage)
    plt.title('Memory Usage Increase with Clustering Features')
    plt.xlabel('Method')
    plt.ylabel('Memory Usage Increase (MB)')
    plt.savefig(os.path.join(save_dir, 'memory_usage_with_clustering.png'))
    plt.show()


def plot_memory_usage(methods, n_components_dict, X_train, X_test, save_dir):
    memory_usage = []
    process = psutil.Process(os.getpid())

    for method, n_components in n_components_dict.items():
        # Get memory usage before applying dimensionality reduction
        mem_before = process.memory_info().rss

        apply_dimensionality_reduction(X_train, X_test, method, n_components)

        # Get memory usage after applying dimensionality reduction
        mem_after = process.memory_info().rss

        # Calculate the absolute increase in memory usage
        mem_increase = abs(mem_after - mem_before) / 1024 ** 2  # Convert to MB
        memory_usage.append(mem_increase)

    # Plotting the memory usage
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(n_components_dict.keys()), y=memory_usage)
    plt.title('Memory Usage')
    plt.xlabel('Method')
    plt.ylabel('Memory Usage(MB)')
    plt.savefig(os.path.join(save_dir, 'memory_usage.png'))
    plt.show()
def print_run_times(run_times, title, save_dir):
    plt.figure(figsize=(10, 6))
    for method, times in run_times.items():
        n_components = [x[0] for x in times]
        elapsed_times = [x[1] for x in times]
        sns.lineplot(x=n_components, y=elapsed_times, label=method)
    plt.title(title)
    plt.xlabel('Number of Components')
    plt.ylabel('Elapsed Time (seconds)')
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

    # Neural network on reduced data with different n_components
    n_components_range = [2, 4, 6, 8, 10, 12, 14, 16]  # Example range for number of components
    nn_results = {}
    run_times = {}
    for method in ['RP', 'PCA', 'ICA', 't-SNE']:
        if method == 't-SNE':
            nn_results[method], run_times[method] = run_nn_experiment_with_components(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, method, [2, 3])
        else:
            nn_results[method], run_times[method] = run_nn_experiment_with_components(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, method, n_components_range)

    # Add baseline accuracy
    baseline_accuracy = 0.9450
    nn_results['Baseline'] = [(n, baseline_accuracy) for n in n_components_range]

    # Plot NN results
    plot_nn_results(nn_results, 'Neural Network on Reduced Data', save_dir)

    # Plot run times
    print_run_times(run_times, 'Run Times for Dimensionality Reduction Methods', save_dir)

    # Neural network with clustering features
    em_clusters = 16  # Example number of clusters for EM
    kmean_clusters = 32  # Example number of clusters for KMeans
    # nn_clustering_results = []
    # nn_clustering_results.append(
    #     ('Mobile + NN', *run_nn_experiment(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile)))
    # nn_clustering_results.append(('Mobile + EM Features + NN',
    #                               *run_nn_with_clustering_features(X_train_mobile, y_train_mobile, X_test_mobile,
    #                                                               y_test_mobile,
    #                                                               'EM', em_clusters)))
    # nn_clustering_results.append(('Mobile + KMeans Features + NN',
    #                               *run_nn_with_clustering_features(X_train_mobile, y_train_mobile, X_test_mobile,
    #                                                               y_test_mobile,
    #                                                               'KMeans', kmean_clusters)))

    # # Print NN with clustering features results
    # print("\nNeural Network with Clustering Features Results:")
    # for result in nn_clustering_results:
    #     print(f"{result[0]}: Accuracy={result[1]:.4f}")
    #
    # # Plot confusion matrices
    # cm_list = [result[2] for result in nn_clustering_results if result[2] is not None]
    # titles = [result[0] for result in nn_clustering_results if result[2] is not None]
    # plot_confusion_matrices(cm_list, titles, save_dir)

    # New experiments
    n_components_dict = {'RP': 8, 'PCA': 8, 'ICA': 8, 't-SNE': 3}

    # Loss curves
    # plot_loss_curves(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, ['RP', 'PCA', 'ICA', 't-SNE'], n_components_dict, save_dir)
    #
    # # Learning curves
    # plot_learning_curves(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, ['RP', 'PCA', 'ICA', 't-SNE'], n_components_dict, save_dir)

    # Memory usage
    plot_memory_usage(['RP', 'PCA', 'ICA', 't-SNE'], n_components_dict, X_train_mobile, X_test_mobile, save_dir)

    # Clustering features experiments
    clustering_methods = ['Mobile + NN', 'Mobile + EM Features + NN', 'Mobile + KMeans Features + NN']
    clustering_results = [run_nn_experiment(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile),
                          run_nn_with_clustering_features(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, 'EM', em_clusters),
                          run_nn_with_clustering_features(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, 'KMeans', kmean_clusters)]

    # Loss curves for clustering features
    plot_loss_curves_with_clustering(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, clustering_methods, {'Mobile + NN': 0, 'Mobile + EM Features + NN': em_clusters, 'Mobile + KMeans Features + NN': kmean_clusters}, save_dir)

    # Learning curves for clustering features
    plot_learning_curves_with_clustering(X_train_mobile, y_train_mobile, X_test_mobile, y_test_mobile, clustering_methods, {'Mobile + NN': 0, 'Mobile + EM Features + NN': em_clusters, 'Mobile + KMeans Features + NN': kmean_clusters}, save_dir)

    # Memory usage for clustering features
    # plot_memory_usage_with_clustering(clustering_methods, {'Mobile + NN': 0, 'Mobile + EM Features + NN': em_clusters, 'Mobile + KMeans Features + NN': kmean_clusters}, X_train_mobile, X_test_mobile, save_dir)

else:
    print("Error: One or both datasets could not be loaded.")
def print_run_times(run_times, title, save_dir):
    plt.figure(figsize=(10, 6))
    for method, times in run_times.items():
        n_components = [x[0] for x in times]
        elapsed_times = [x[1] for x in times]
        sns.lineplot(x=n_components, y=elapsed_times, label=method)
    plt.title(title)
    plt.xlabel('Number of Components')
    plt.ylabel('Elapsed Time (seconds)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{title}.png'))
    plt.show()
