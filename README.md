# Machine Learning Projects 

This repository contains a collection of machine learning projects covering core concepts such as web data extraction, concept learning, clustering techniques, and binary text classification. Each project is implemented in Python using libraries like `scikit-learn`, `matplotlib`, and `pandas`.

---

## Project List

### Web Scraping
Extracts and processes textual content from web pages to prepare datasets for machine learning models. This project demonstrates how to collect unstructured data and clean it for downstream analysis.

- **Tech stack:** `BeautifulSoup`, `requests`, `pandas`
- **Key tasks:** HTML parsing, content filtering, text cleaning

---

### Candidate Elimination Method
Implements the **Candidate Elimination Algorithm** to learn hypotheses consistent with training data. This project illustrates concept learning using version spaces (specific and general boundaries).

- **Tech stack:** Python 
- **Key concept:** Consistent hypothesis space generation from labeled examples

---

### K-Means Clustering
Performs unsupervised clustering using the K-Means algorithm. The project explores how data points are grouped into `k` clusters based on feature similarity, and visualizes the results.

- **Tech stack:** `scikit-learn`, `matplotlib`
- **Highlights:** Centroid initialization, cluster iteration, elbow method visualization

---

### Hierarchical Clustering
Applies **Agglomerative Hierarchical Clustering** using both **single** and **complete** linkage methods. Dendrograms are used to visualize cluster formation at different distance thresholds.

- **Tech stack:** `scipy`, `matplotlib`
- **Input:** Condensed pairwise distance matrix
- **Output:** Dendrograms with labeled data points

---

### Real vs. Fake News Classification
Classifies news articles as real or fake using supervised learning models. The dataset includes pre-labeled real and fake news articles. Text data is vectorized and evaluated using multiple ML algorithms.

- **Preprocessing:** TF-IDF vectorization, label encoding, PCA (2 components)
- **Models Used:** 
  - Gaussian, Multinomial & Bernoulli Naïve Bayes  
  - Neural Network (with grid search optimization)  
  - SVM (linear and RBF kernels)
- **Evaluation Methods:**
  - 70/30 Train-Test (Random & Stratified)
  - 10-Fold Cross-Validation
  - Accuracy & F1 Score

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn scipy beautifulsoup4 requests
