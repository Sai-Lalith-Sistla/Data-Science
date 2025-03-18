# Regression Study Notes
## **1. What is Regression?**
Regression is a supervised learning technique used to model the relationship between a dependent variable (target) and one or more independent variables (predictors). It is used to predict continuous numerical values based on input features.

### **Key Concepts:**
| Term                 | Description |
|----------------------|-------------|
| **Dependent Variable (Y)** | The target/output variable to be predicted. |
| **Independent Variables (X1, X2,…Xn)** | The features influencing the target. |
| **Regression Coefficients** | The weights assigned to features to determine their impact on the target. |
| **Loss Function** | Measures the error in prediction (e.g., Mean Squared Error, Mean Absolute Error). |

---

## **2. Difference between Regression & Classification**
| Feature          | Regression                                   | Classification |
|-----------------|---------------------------------|----------------|
| **Definition**  | Predicts continuous values.     | Predicts categorical labels. |
| **Output Type** | Numeric (e.g., sales figures).  | Discrete classes (e.g., spam/not spam). |
| **Example Models** | Linear Regression, Ridge, Lasso, SVR, Neural Networks (for continuous output). | Logistic Regression, Decision Trees, SVM, Neural Networks (for classification). |
| **Evaluation Metrics** | RMSE, MSE, R² Score. | Accuracy, Precision, Recall, F1-score. |

---

## **3. Real-World Applications of Regression**

| Application         | Description |
|--------------------|-------------|
| **Sales Prediction** | Used in **Retail and E-commerce** to forecast future revenue. Features include past sales, seasonality, customer demographics, and promotions. |
| **Risk Analysis** | Used in **Finance & Insurance** to predict risk scores based on credit history, transaction patterns, and economic factors. |
| **Demand Forecasting** | Applied in **Supply Chain & Manufacturing** to predict product demand based on historical sales, pricing, and economic indicators. |

---

## **4. Types of Regression Algorithms (Overview)**

| Regression Type | Description | Use Case |
|----------------|-------------|----------|
| **Linear Regression** | Assumes a linear relationship between dependent and independent variables. | Basic sales predictions, stock price forecasting. |
| **Polynomial Regression** | Extends Linear Regression by adding polynomial terms (higher-degree features). | Complex, non-linear relationships (e.g., predicting housing prices). |
| **Ridge & Lasso Regression** | Regularized regression models that prevent overfitting (Ridge uses L2, Lasso uses L1 regularization). | High-dimensional datasets (e.g., genomic data analysis). |
| **Logistic Regression** | Despite the name, it is used for classification rather than regression. | Spam detection, fraud detection. |
| **Support Vector Regression (SVR)** | Uses SVMs for regression by finding the best-fit hyperplane within a threshold. | Time-series forecasting, financial modeling. |
| **Decision Tree Regression** | Splits the dataset into hierarchical decisions based on feature values. | Predicting house prices, loan default prediction. |
| **Random Forest Regression** | Uses an ensemble of decision trees to improve accuracy and reduce variance. | Complex real-world problems like energy consumption forecasting. |
| **Gradient Boosting Regression (XGBoost, LightGBM, CatBoost)** | Uses boosting techniques to iteratively improve predictions. | Kaggle competitions, stock market predictions. |
| **Neural Networks for Regression** | Uses deep learning architectures (e.g., MLPs, CNNs, LSTMs) to model complex patterns. | Predicting sales based on images, customer churn forecasting. |


## **5. Metrics to Measure Performance**

| Metric | Description | When to Use? |
|--------|------------|-------------|
| **Mean Squared Error (MSE)** | Measures average squared differences between predicted and actual values. | Suitable when large errors need to be heavily penalized. |
| **Root Mean Squared Error (RMSE)** | Square root of MSE, more interpretable in original units. | Used when we want to understand error magnitude. |
| **Mean Absolute Error (MAE)** | Measures average absolute differences between predicted and actual values. | Suitable when all errors should be treated equally. |
| **R-Squared (R² Score)** | Represents the proportion of variance explained by the model. | Used when we need to measure how well independent variables explain the target. |
| **Adjusted R-Squared** | Adjusts R² for the number of predictors, preventing overestimation. | Suitable for multiple regression with multiple independent variables. |
| **Mean Absolute Percentage Error (MAPE)** | Measures the percentage error in predictions. | Used when comparing error across datasets with different scales. |

---

## **6. Detailed Metric Information**

| Metric Name | Formula / How Derived | How to Read Values | Libraries for Calculation |
|------------|-----------------------|---------------------|-------------------------|
| **MSE** | \( \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \) | Lower is better. High values indicate large errors. | `sklearn.metrics.mean_squared_error()` |
| **RMSE** | \( \sqrt{MSE} \) | Lower is better. Easier to interpret than MSE. | `numpy.sqrt(MSE)`, `sklearn.metrics.mean_squared_error(squared=False)` |
| **MAE** | \( \frac{1}{n} \sum |y_i - \hat{y}_i| \) | Lower is better. Less sensitive to large errors than MSE. | `sklearn.metrics.mean_absolute_error()` |
| **R² Score** | \( 1 - \frac{SS_{res}}{SS_{tot}} \) | Higher is better (max 1). Measures variance explained. | `sklearn.metrics.r2_score()` |
| **Adjusted R²** | \( 1 - \frac{(1-R^2)(n-1)}{n-p-1} \) | Higher is better, accounts for number of predictors. | Manually calculated or `statsmodels.api.OLS().rsquared_adj` |
| **MAPE** | \( \frac{1}{n} \sum \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100 \) | Lower is better. Expressed as a percentage error. | Custom implementation or `sklearn.metrics.mean_absolute_percentage_error()` |


## **7. Data Preprocessing Steps for Regression Algorithms**

| Preprocessing Step              | Description | Must/Optional |
|---------------------------------|-------------|--------------|
| **Handling Missing Values**     | Fill or drop missing values using mean, median, or interpolation. | Must |
| **Feature Scaling**             | Normalize or standardize features (e.g., MinMaxScaler, StandardScaler). | Must for algorithms like Ridge, Lasso, SVR, and Neural Networks |
| **Encoding Categorical Variables** | Convert categorical data into numerical format (One-Hot Encoding, Label Encoding). | Must if categorical features are present |
| **Feature Selection**           | Remove irrelevant features to improve model performance. | Optional (but recommended for high-dimensional data) |
| **Outlier Detection & Handling** | Detect and treat outliers using IQR, Z-score, or transformations. | Optional but recommended for robust models |
| **Multicollinearity Check**      | Check if independent variables are highly correlated. | Optional, but important for Linear Regression |
| **Polynomial Feature Engineering** | Add polynomial terms for better non-linear relationships. | Optional, needed for Polynomial Regression |
| **Dimensionality Reduction**     | Use PCA, LDA, or Feature Selection techniques for high-dimensional data. | Optional, useful for complex models |

---

## **8. Assumptions of Regression Algorithms**

### **Linear Regression Assumptions**
1. **Linearity:** The relationship between independent and dependent variables should be linear.
2. **Homoscedasticity:** Constant variance of residuals.
3. **Independence:** Observations should be independent.
4. **Normality of Residuals:** Residuals should be normally distributed.
5. **No Multicollinearity:** Independent variables should not be highly correlated.

#### **Code to Check Assumptions**
```python
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('your_data.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']
X = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary())

# Check normality of residuals
sns.histplot(model.resid, kde=True)
```

---

### **Ridge & Lasso Regression Assumptions**
- Similar to Linear Regression but designed to handle multicollinearity.
- Requires feature scaling.

#### **Code to Check for Multicollinearity**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
```

---

### **Decision Tree & Random Forest Regression Assumptions**
- No strict assumptions about data distribution.
- Handles categorical variables and non-linearity well.
- Prone to overfitting; pruning and hyperparameter tuning are needed.

#### **Code to Tune Hyperparameters**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf = RandomForestRegressor()
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X, y)
print(grid_search.best_params_)
```

---

### **Support Vector Regression (SVR) Assumptions**
- Requires feature scaling.
- Works well for non-linear problems using kernels.

#### **Feature Scaling Code**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### **Neural Networks for Regression Assumptions**
- Requires large data.
- Sensitive to feature scaling.
- Works best with hyperparameter tuning.

#### **Code for Feature Scaling & Model Training**
```python
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)
```

---





