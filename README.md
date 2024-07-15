# CS7641 Assignment 3 Project Code

## Project Overview

This project is for the third assignment of the CS7641 course in the Summer 2024 semester, focusing on Unsupervised Learning and Dimensionality Reduction. The assignment requires the implementation and analysis of various clustering and dimensionality reduction algorithms, and exploring their performance on different datasets.

## Project Structure

```
CS7641_Assignment3/
│
├── data/                      # Folder for datasets
│
├── images/                    # Folder for storing generated plots
│
├── src/                       # Source code folder
│   ├── time_count_RP_PCA_ICA_tSNE.py    # Script for timing and analyzing RP, PCA, ICA, and t-SNE
│   ├── analyze_RP_PCA_ICA.py            # Script for analyzing RP, PCA, and ICA results
│   ├── time_count_EM_Kmeans.py          # Script for timing and analyzing EM and KMeans
│   ├── time_count_NN.py                 # Script for timing and analyzing Neural Networks
│   ├── main.py                          # Main script to run experiments 1, 2, 3
│   ├── main_nn.py                       # Main script to run Neural Network experiments 4, 5
│
└── README.md                  # Project description file
```

## Prerequisite Libraries

Before running the code, please ensure the following Python libraries are installed:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Usage Instructions

1. Ensure the datasets are placed in the `data/` folder.
2. Run the `src/main.py` script to execute all algorithms and generate results.
3. Run the `src/main_nn.py` script to execute Neural Network related experiments.
4. The generated plots and results will be saved in the `images/` folder.

## Code Description

### `time_count_RP_PCA_ICA_tSNE.py`
Script for timing and analyzing Random Projection (RP), Principal Component Analysis (PCA), Independent Component Analysis (ICA), and t-SNE.

### `analyze_RP_PCA_ICA.py`
Script for analyzing the results of RP, PCA, and ICA.

### `time_count_EM_Kmeans.py`
Script for timing and analyzing Expectation Maximization (EM) and KMeans clustering.

### `time_count_NN.py`
Script for timing and analyzing Neural Networks.

### `main.py`
Main script to run experiments 1, 2, 3.

### `main_nn.py`
Main script to run Neural Network related experiments 4, 5.