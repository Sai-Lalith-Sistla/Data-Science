# Classification Study Notes (Intermediate/Advanced)

## **1. What is Classification?**
Classification is a supervised learning technique used to predict categorical labels based on input features. The output is discrete and belongs to predefined classes.

### **Key Concepts:**
| Term                 | Description |
|----------------------|-------------|
| **Dependent Variable (Y)** | The target/output variable to be classified. |
| **Independent Variables (X1, X2,…Xn)** | The features influencing the classification. |
| **Decision Boundary** | A boundary separating different classes in feature space. |
| **Loss Function** | Measures the error in classification (e.g., Cross-Entropy Loss, Hinge Loss). |

---

## **2. Difference between Classification & Regression**
| Feature          | Classification                                 | Regression |
|-----------------|---------------------------------|----------------|
| **Definition**  | Predicts categorical labels.     | Predicts continuous values. |
| **Output Type** | Discrete (e.g., spam/not spam).  | Numeric (e.g., sales figures). |
| **Example Models** | Logistic Regression, Decision Trees, SVM, Neural Networks (for classification). | Linear Regression, Ridge, Lasso, SVR, Neural Networks (for continuous output). |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-score. | RMSE, MSE, R² Score. |

---

## **3. Real-World Applications of Classification**

| Application         | Description |
|--------------------|-------------|
| **Spam Detection** | Classifies emails as spam or not spam based on textual features. |
| **Fraud Detection** | Identifies fraudulent transactions using behavioral patterns. |
| **Medical Diagnosis** | Predicts diseases based on patient symptoms and test results. |
| **Sentiment Analysis** | Classifies user reviews as positive, neutral, or negative. |
| **Image Recognition** | Identifies objects, animals, or people in images. |

---

## **4. Types of Classification Algorithms (Overview)**

| Classification Type | Description | Use Case |
|----------------|-------------|----------|
| **Logistic Regression** | A probabilistic model that uses a sigmoid function to classify data. | Binary classification tasks like fraud detection. |
| **K-Nearest Neighbors (KNN)** | Classifies based on the majority vote of K nearest neighbors. | Image recognition, recommendation systems. |
| **Decision Tree Classifier** | Splits the dataset into hierarchical decisions based on feature values. | Customer segmentation, risk assessment. |
| **Random Forest Classifier** | Uses an ensemble of decision trees for better accuracy. | Medical diagnosis, spam detection. |
| **Support Vector Machine (SVM)** | Finds the optimal hyperplane to separate classes. | Text classification, facial recognition. |
| **Naïve Bayes Classifier** | Based on Bayes’ theorem, assuming independence among features. | Spam filtering, sentiment analysis. |
| **Gradient Boosting Classifiers (XGBoost, LightGBM, CatBoost)** | Boosting-based ensemble models for high accuracy. | Kaggle competitions, click-through rate prediction. |
| **Neural Networks for Classification** | Uses deep learning architectures (e.g., CNNs, LSTMs) for feature extraction. | Image classification, speech recognition. |

---

## **5. Metrics to Measure Performance**

| Metric | Description | When to Use? |
|--------|------------|-------------|
| **Accuracy** | Measures the percentage of correct predictions. | When classes are balanced. |
| **Precision** | Measures the proportion of true positives among predicted positives. | When false positives are costly (e.g., fraud detection). |
| **Recall (Sensitivity)** | Measures the proportion of true positives correctly identified. | When false negatives are costly (e.g., medical diagnosis). |
| **F1-Score** | Harmonic mean of precision and recall. | When both false positives and false negatives matter. |
| **ROC-AUC Score** | Measures the model’s ability to distinguish between classes. | When evaluating probability-based classifiers. |
| **Log Loss (Cross-Entropy Loss)** | Measures the uncertainty of predictions. | When working with probabilistic models. |

---

## **6. Detailed Metric Information**

| Metric Name | Formula / How Derived | How to Read Values | Libraries for Calculation |
|------------|-----------------------|---------------------|-------------------------|
| **Accuracy** | \( \frac{TP + TN}{TP + TN + FP + FN} \) | Higher is better. Measures overall correctness. | `sklearn.metrics.accuracy_score()` |
| **Precision** | \( \frac{TP}{TP + FP} \) | Higher is better. Focuses on reducing false positives. | `sklearn.metrics.precision_score()` |
| **Recall** | \( \frac{TP}{TP + FN} \) | Higher is better. Focuses on reducing false negatives. | `sklearn.metrics.recall_score()` |
| **F1-Score** | \( 2 \times \frac{Precision \times Recall}{Precision + Recall} \) | Higher is better. Balance between precision and recall. | `sklearn.metrics.f1_score()` |
| **ROC-AUC** | Based on the area under the ROC curve. | Higher is better. Evaluates the model’s ranking ability. | `sklearn.metrics.roc_auc_score()` |
| **Log Loss** | \( -\frac{1}{N} \sum (y \log \hat{y} + (1 - y) \log (1 - \hat{y})) \) | Lower is better. Measures classification confidence. | `sklearn.metrics.log_loss()` |

---

## **7. Data Preprocessing for Classification Algorithms**

| Preprocessing Step | Description | Must/Optional |
|------------------|-------------|--------------|
| **Handling Missing Values** | Fill or drop missing values using mean, median, or mode. | Must |
| **Feature Scaling** | Normalize or standardize features. | Must for SVM, KNN, Neural Networks |
| **Encoding Categorical Variables** | Convert categorical data into numerical format. | Must if categorical features are present |
| **Handling Imbalanced Data** | Use oversampling, undersampling, or class weighting. | Must if dataset is imbalanced |
| **Feature Selection** | Remove irrelevant features to improve model performance. | Optional but recommended |

---

## **8. Assumptions of Classification Algorithms**

### **Logistic Regression Assumptions**
- Linearity of independent variables with log-odds.
- No multicollinearity.
- Independent observations.

#### **Code to Check Multicollinearity**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
```

---

### **SVM Assumptions**
- Data should be scaled.
- Works best when data is separable.

#### **Feature Scaling Code**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### **Decision Tree & Random Forest Assumptions**
- No strict assumptions about data distribution.
- Handles categorical variables well.

#### **Hyperparameter Tuning Code**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X, y)
print(grid_search.best_params_)
```

---



