# Regularization Study Notes

## **1. What is Regularization?**
Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the loss function. It helps in simplifying models and improving generalization to unseen data.

### **Why is Regularization Needed?**
- Prevents overfitting by discouraging overly complex models.
- Enhances model generalization to new data.
- Reduces variance without significantly increasing bias.

---

## **2. Types of Regularization Techniques**

| Regularization Type | Description | Use Case |
|-----------------|-------------|----------|
| **L1 Regularization (Lasso Regression)** | Adds the absolute value of coefficients as a penalty term \( \lambda \sum |w| \) | Feature selection, sparse models |
| **L2 Regularization (Ridge Regression)** | Adds the squared values of coefficients as a penalty term \( \lambda \sum w^2 \) | Preventing overfitting while keeping all features |
| **Elastic Net Regularization** | Combines L1 and L2 regularization \( \lambda_1 \sum |w| + \lambda_2 \sum w^2 \) | When both feature selection and weight shrinkage are needed |
| **Dropout (for Neural Networks)** | Randomly drops neurons during training to prevent over-reliance on certain features. | Deep learning models |
| **Early Stopping** | Stops training when validation loss starts increasing. | Neural networks, gradient boosting models |
| **Batch Normalization** | Normalizes inputs of each layer to reduce internal covariate shift. | Deep learning models |
| **Data Augmentation** | Increases dataset diversity to reduce overfitting. | Image and text-based models |

---

## **3. Real-World Applications of Regularization**

| Application | Description |
|------------|-------------|
| **Feature Selection** | Lasso regression removes irrelevant features, making models more interpretable. |
| **Text Classification** | Ridge regression prevents overfitting when using high-dimensional text features. |
| **Deep Learning** | Dropout is commonly used in CNNs and RNNs to improve generalization. |
| **Finance** | Regularization helps in reducing noise and making stable predictions. |
| **Healthcare** | Ensures ML models generalize well in medical diagnosis tasks. |

---

## **4. Metrics to Measure Regularization Performance**

| Metric | Description | When to Use? |
|--------|------------|-------------|
| **Mean Squared Error (MSE)** | Measures average squared error between predictions and actual values. | Regression models with L2 regularization. |
| **Mean Absolute Error (MAE)** | Measures absolute differences, useful when errors need equal weighting. | Regression models with L1 regularization. |
| **R² Score** | Indicates how well the model explains variance in the data. | Checking overall model performance. |
| **Cross-Validation Score** | Measures model performance across different data subsets. | Evaluating regularization impact. |
| **Log Loss** | Measures confidence in classification predictions. | Logistic regression models with L1/L2 regularization. |

---

## **5. Regularization Hyperparameters & Tuning**

| Hyperparameter | Regularization Type | Effect |
|---------------|--------------------|--------|
| **Lambda (\( \lambda \))** | Lasso, Ridge, Elastic Net | Controls the strength of regularization. Higher values shrink coefficients more. |
| **Alpha (\( \alpha \))** | Elastic Net | Balances L1 vs L2 regularization (0 = Ridge, 1 = Lasso). |
| **Dropout Rate** | Dropout Regularization | Controls the percentage of neurons dropped during training. |
| **Early Stopping Patience** | Early Stopping | Determines how many epochs to wait before stopping training. |
| **Batch Norm Momentum** | Batch Normalization | Controls how much past batches influence normalization statistics. |

---

## **6. Data Preprocessing for Regularization Techniques**

| Preprocessing Step | Description | Must/Optional |
|------------------|-------------|--------------|
| **Feature Scaling** | Standardize or normalize data before applying regularization. | Must for Lasso, Ridge, Elastic Net, SVM, and Neural Networks. |
| **Handling Missing Values** | Fill missing values before training. | Must if dataset has NaN values. |
| **Feature Engineering** | Create meaningful interactions before applying L1/L2 regularization. | Optional but recommended. |
| **Train-Test Splitting** | Ensure model generalization before applying regularization. | Must |
| **Encoding Categorical Features** | Convert categorical data to numeric values before applying regularization. | Must |

---

## **7. Assumptions of Regularization Techniques**

### **L1 & L2 Regularization (Lasso & Ridge) Assumptions:**
- Features should be standardized.
- Works best when multicollinearity is present.
- Lasso is effective when sparsity is needed (many features should be zero).

#### **Code to Apply Ridge & Lasso Regularization:**
```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)  # L2 Regularization
lasso = Lasso(alpha=0.1)  # L1 Regularization

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
```

---

### **Elastic Net Assumptions:**
- Useful when Lasso selects too few features and Ridge keeps all features.
- Requires careful tuning of the \( \alpha \) parameter.

#### **Code to Apply Elastic Net Regularization:**
```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
```

---

### **Dropout Regularization Assumptions:**
- Applied only to deep learning models.
- Works best when used with batch normalization.

#### **Code to Apply Dropout in Neural Networks:**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.5),  # Drops 50% of neurons randomly
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## **8. Choosing the Right Regularization Technique**

| Scenario | Recommended Regularization |
|----------|---------------------------|
| **High-dimensional data with irrelevant features** | Lasso (L1 Regularization) |
| **Multicollinearity in features** | Ridge (L2 Regularization) |
| **Need for a balance between L1 & L2** | Elastic Net |
| **Deep learning models** | Dropout Regularization |
| **Preventing early overfitting** | Early Stopping |
| **Ensuring stable weight distributions** | Batch Normalization |



