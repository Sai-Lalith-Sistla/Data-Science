# Principal Component Analysis (PCA) Study Notes 

## **1. What is PCA?**
Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and statistics. It transforms a high-dimensional dataset into a lower-dimensional one while retaining the most important variance in the data.

### **Why is PCA Needed?**
- Reduces computational complexity by lowering the number of features.
- Helps in visualization of high-dimensional data.
- Mitigates the curse of dimensionality.
- Removes multicollinearity among features.
- Improves model performance by reducing noise.

---

## **2. How PCA Works?**

| Step | Description |
|------|-------------|
| **1. Standardization** | Data is standardized to have zero mean and unit variance. |
| **2. Covariance Matrix Computation** | Measures relationships between variables. |
| **3. Eigenvalues & Eigenvectors** | Extracts principal components that capture variance. |
| **4. Sorting & Selection** | Selects top components that explain the most variance. |
| **5. Projection** | Transforms data into the new reduced feature space. |

---

## **3. Real-World Applications of PCA**

| Application | Description |
|------------|-------------|
| **Image Compression** | Reduces the size of image datasets while preserving essential information. |
| **Finance & Risk Analysis** | Identifies key factors driving stock market trends. |
| **Gene Expression Analysis** | Finds significant genes in medical research. |
| **Anomaly Detection** | Reduces dimensionality for efficient fraud detection. |
| **Recommendation Systems** | Improves efficiency by reducing sparse high-dimensional data. |

---

## **4. Metrics to Measure PCA Performance**

| Metric | Description | When to Use? |
|--------|------------|-------------|
| **Explained Variance Ratio** | Shows how much variance each principal component captures. | Evaluating how many components to retain. |
| **Reconstruction Error** | Measures loss of information after PCA transformation. | Checking if too much information is lost. |
| **Cumulative Variance** | Sum of variance explained by selected components. | Ensuring retained components capture sufficient variance. |

---

## **5. Choosing the Right Number of Principal Components**

### **Variance Threshold Approach:**
- Choose components that retain at least 95% of the total variance.
- Plot cumulative variance and determine the elbow point.

#### **Code to Find Optimal Number of Components:**
```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

pca = PCA().fit(X_train)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
```

---

## **6. Data Preprocessing for PCA**

| Preprocessing Step | Description | Must/Optional |
|------------------|-------------|--------------|
| **Feature Scaling** | Standardize data before applying PCA. | Must |
| **Handling Missing Values** | Fill missing values to avoid distortions. | Must |
| **Removing Highly Correlated Features** | Redundant features can skew PCA results. | Optional |

---

## **7. Assumptions of PCA**

| Assumption | Description |
|-----------|-------------|
| **Linearity** | PCA assumes linear relationships between features. |
| **High Variance Importance** | Assumes that the most important information is captured in high variance. |
| **Noisy Data Impact** | Sensitive to outliers and noise. |

#### **Code to Apply PCA in Python:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

---

## **8. Choosing Between PCA and Other Dimensionality Reduction Techniques**

| Scenario | Recommended Method |
|----------|-------------------|
| **Linear relationships between features** | PCA |
| **Non-linear patterns in data** | t-SNE, UMAP |
| **Feature selection over transformation** | Lasso Regression, Random Forest Feature Importance |
| **Sparse datasets** | Truncated SVD |

---