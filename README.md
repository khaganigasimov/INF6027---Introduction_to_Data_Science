Stock Price Prediction - INF6027

Overview

This project explores stock price prediction using financial indicators and machine learning models. The dataset consists of financial data from publicly traded US companies between 2014 and 2018, providing 200+ financial indicators per stock. The primary objective is to classify stocks as 'Buy' or 'Not Buy' and to predict stock price variation using regression models.

Project Structure

data/: Contains raw and processed datasets.

notebooks/: Jupyter and R Notebooks for data analysis and modeling.

src/: Scripts for data preprocessing, feature selection, and model training.

reports/: Research methodology, results, and discussions.

figures/: Visualizations and analysis graphs.

README.md: Project overview and instructions.

requirements.txt: List of dependencies.

.gitignore: Files to be ignored in version control.

Methodology

Data Preprocessing

Handled missing values using median imputation.

Identified and addressed outliers using the IQR method.

Merged datasets from different years into a unified dataframe.

Feature Selection & Dimensionality Reduction

Classification (Information Value Approach): Retained 138 features with significant predictive power.

Regression (Two Approaches):

Mutual Information & Multicollinearity Removal: Selected the top 89 features.

Principal Component Analysis (PCA): Retained 120 principal components explaining 92.5% of variance.

Modeling

Classification: Logistic Regression to classify stocks as 'Buy' or 'Not Buy'.

Regression: Linear Regression for stock price variation prediction.

Model evaluation through accuracy, R-squared, MSE, and RMSE.

Results

Classification Accuracy: 63.45%

Regression R-squared: ~20%

Feature Importance Analysis:

Top financial indicators affecting stock price variation.

PCA significantly reduced dimensionality but lost interpretability.

Mutual Information preserved interpretability but required multicollinearity handling.

Dependencies

Install required packages using:

pip install -r requirements.txt

How to Run

Clone the repository:

git clone https://github.com/your-username/Stock_Price_Prediction_INF6027.git
cd Stock_Price_Prediction_INF6027

Run Jupyter Notebook:

jupyter notebook

Execute notebooks in sequence for data preprocessing, feature selection, and modeling.

Contributors

Your Name

License

This project is licensed under the MIT License.
