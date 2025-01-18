## Stock Price Prediction - INF6027

# Overview

This project explores stock price prediction using financial indicators and machine learning models. The dataset consists of financial data from publicly traded US companies between 2014 and 2018, providing 200+ financial indicators per stock. The primary objective is to classify stocks as 'Buy' or 'Not Buy' and to predict stock price variation using regression models.

# Project Structure

data/: Contains raw and processed datasets.

notebooks/: R Notebooks for data analysis and modeling.

README.md: Project overview and instructions.


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

# Feature Importance Analysis:

Top financial indicators affecting stock price variation.

PCA significantly reduced dimensionality but lost interpretability.

Mutual Information preserved interpretability but required multicollinearity handling.

# Technologies used 

R proogramming language with following libraries:
ggplot2, infotheo, dplyr, caret, and Information.

# How to Run

Install the latest version of R and RStudio:
Download R from the CRAN website.
Download and install RStudio from the RStudio website.

Clone the repository:

git clone https://github.com/your-username/Stock_Price_Prediction_INF6027.git
cd Stock_Price_Prediction_INF6027
