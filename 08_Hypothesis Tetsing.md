# Hypothesis Testing Notes

## Introduction
Hypothesis testing is a statistical method used to make decisions based on data. It helps determine whether there is enough evidence to reject a null hypothesis in favor of an alternative hypothesis.

## Key Concepts

### 1. Null Hypothesis (H₀)
- Represents the default assumption or status quo.
- Example: "There is no significant difference between two sample means."

### 2. Alternative Hypothesis (H₁ or Ha)
- Represents the claim that contradicts H₀.
- Example: "There is a significant difference between two sample means."

### 3. Significance Level (α)
- The probability of rejecting H₀ when it is actually true (Type I Error).
- Common values: 0.05 (5%), 0.01 (1%), 0.10 (10%).

### 4. P-Value
- The probability of obtaining test results at least as extreme as the observed results, assuming H₀ is true.
- If **p-value < α**, reject H₀.

### 5. Type I and Type II Errors
- **Type I Error (False Positive)**: Rejecting H₀ when it is true.
- **Type II Error (False Negative)**: Failing to reject H₀ when it is false.

### 6. Power of a Test
- Probability of correctly rejecting H₀ when H₁ is true.
- **Power = 1 - Probability of Type II Error.**

## Common Hypothesis Tests and Interpretation

| Test Name | Description | Interpretation |
|-----------|------------|---------------|
| **Z-Test** | Used for large sample sizes (n > 30) and known population variance. Example: Comparing sample mean with population mean. | If the computed Z-score falls beyond the critical value (or p-value < α), reject H₀; otherwise, fail to reject H₀. |
| **T-Test** | Used for small samples (n < 30) and unknown population variance. Types: One-sample t-test (sample vs. population), Two-sample t-test (two independent groups), Paired t-test (before and after treatment). | If the t-statistic exceeds the critical value or p-value < α, reject H₀; otherwise, fail to reject H₀. |
| **Chi-Square Test** | Used for categorical data to test relationships between variables. Types: Goodness-of-Fit Test (expected vs. observed distribution), Test for Independence (relationship between two categorical variables). | If the chi-square statistic is greater than the critical value or p-value < α, reject H₀; otherwise, fail to reject H₀. |
| **ANOVA (Analysis of Variance)** | Compares means of three or more groups. Types: One-way ANOVA (one independent variable), Two-way ANOVA (two independent variables). | If the F-statistic is greater than the critical value or p-value < α, reject H₀, indicating at least one group mean is different. |
| **Mann-Whitney U Test** | Non-parametric test for comparing two independent groups when assumptions of t-test are violated. | If the U-statistic is lower than the critical value or p-value < α, reject H₀, indicating a significant difference between groups. |
| **Wilcoxon Signed-Rank Test** | Non-parametric equivalent of paired t-test for dependent samples. | If the test statistic is beyond the critical value or p-value < α, reject H₀, indicating a significant difference. |
| **Kruskal-Wallis Test** | Non-parametric equivalent of ANOVA for comparing three or more independent groups. | If the H-statistic is larger than the critical value or p-value < α, reject H₀, suggesting at least one group differs. |
| **A/B Testing** | Used in business and web analytics to compare two versions of a product or webpage. Often uses a t-test or chi-square test for statistical validation. | If the test statistic exceeds the critical value or p-value < α, reject H₀, suggesting a significant difference between versions. |

## Steps in Hypothesis Testing
1. **Define H₀ and H₁.**
2. **Choose significance level (α).**
3. **Select an appropriate test and check assumptions.**
4. **Compute the test statistic.**
5. **Find the p-value or compare with the critical value.**
6. **Make a decision (Reject or Fail to Reject H₀).**
7. **Draw conclusions and interpret results.**

## Conclusion
Hypothesis testing is a fundamental statistical tool for making data-driven decisions. Choosing the right test and correctly interpreting results is crucial for drawing meaningful conclusions.

