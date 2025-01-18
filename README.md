# Stock Price Prediction - INF6027

# Overview

This project explores stock price prediction using financial indicators and machine learning models. The dataset consists of financial data from publicly traded US companies between 2014 and 2018, providing 200+ financial indicators per stock. The primary objective is to classify stocks as 'Buy' or 'Not Buy' and to predict stock price variation using regression models.

# Project Structure

data/: Contains raw and processed datasets.

notebooks/: R Notebooks for data analysis and modeling.

README.md: Project overview and instructions.

# Research Questions

1. Key Financial Indicators: What financial indicators significantly influence stock price variation (PRICE_VAR) and stock classification (Buy/Not Buy, Class variable)?
2. Classification vs. Regression: Given the dual nature of the problem, which approach—classification or regression—provides a more nuanced understanding of stock performance?
3. Impact of Dimensionality Reduction: How does using Principal Component Analysis (PCA) affect the predictive accuracy and interpretability of models in stock price forecasting?
4. Performance Comparison: How do linear models (Linear Regression, Logistic Regression) compare with non-linear models (e.g., ANN, XGBoost) for forecasting stock price variation and classification?

# Methodology

# Data Preprocessing

Handled missing values using median imputation.

Identified and addressed outliers using the IQR method.

Merged datasets from different years into a unified dataframe.

# Feature Selection & Dimensionality Reduction

Classification (Information Value Approach): Retained 138 features with significant predictive power.

Regression (Two Approaches):

Mutual Information & Multicollinearity Removal: Selected the top 89 features.

Principal Component Analysis (PCA): Retained 120 principal components explaining 92.5% of variance.

# Modeling

Classification: Logistic Regression to classify stocks as 'Buy' or 'Not Buy'.

Regression: Linear Regression for stock price variation prediction.

Model evaluation through accuracy, R-squared, MSE, and RMSE.

# Results

Classification Accuracy: 63.45%

Regression R-squared: ~20%

# Key Findings

# Stock Classification

Classification provided a more actionable framework for investors, offering a clear Buy/Not Buy decision.
Information Value was used for feature selection, reducing 200+ features to 138, and further refined through multicollinearity filtering, resulting in 103 final predictors.
Logistic regression achieved 63.45% accuracy, despite the dataset's high dimensionality and complexity.

# Stock Price Regression

Two regression strategies were tested:
PCA-based Regression: PCA reduced dimensionality while retaining 92.5% of variance, though it lacked interpretability.
Mutual Information & Multicollinearity Filtering: Retained interpretability while reducing feature redundancy.
Regression models had limited predictive power (R² ~ 20%), highlighting the difficulty in predicting stock prices using only financial ratios.

# Comparative Insights

Classification is more interpretable and actionable than regression, making it more suitable for investment strategy development.
Dimensionality reduction techniques improve computational efficiency but must be balanced against interpretability.
Stock price forecasting remains inherently complex, suggesting the need for non-linear and deep learning models for future improvements.


# Feature Importance Analysis:

Top financial indicators affecting stock price variation.

PCA significantly reduced dimensionality but lost interpretability.

Mutual Information preserved interpretability but required multicollinearity handling.

# Technologies used 

R proogramming language with following libraries:
ggplot2, infotheo, dplyr, caret, and Information.

# How to Run

**Install the latest version of R and RStudio:**
   - Download R from the [CRAN website](https://cran.r-project.org/)
   - Download and install RStudio from the [RStudio website](https://posit.co/download/rstudio-desktop/)

**Clone the repository:**

git clone https://github.com/your-username/Stock_Price_Prediction_INF6027.git

**Specific points about code**

Inside the Code folder, there are two R files containing the code for Classification and Regression respectively. Data preprocessing parts are the same, differences begin with Feature engineering part. Code can be executed at once or sequentially to see all the details.
