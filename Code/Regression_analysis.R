install.packages("ggplot2")
install.packages("infotheo", dependencies=TRUE)
install.packages("dplyr", dependencies=TRUE)
install.packages("caret")
install.packages("Information")

library(Information)
library(caret)
library(ggplot2)  # For visualization
library(infotheo)
library(dplyr)


# Reading CSV files
df1 <- read.csv("data/2014_Financial_Data.csv")
df2 <- read.csv("data/2015_Financial_Data.csv")
df3 <- read.csv("data/2016_Financial_Data.csv")
df4 <- read.csv("data/2017_Financial_Data.csv")
df5 <- read.csv("data/2018_Financial_Data.csv")

# Display the first few rows of the first dataset
head(df1)


# Check if all datasets have the same column names
all_columns <- list(
  colnames(df1),
  colnames(df2),
  colnames(df3),
  colnames(df4),
  colnames(df5)
)


# Check if all columns are identical
identical_columns <- all(sapply(all_columns, function(x) identical(x, colnames(df1))))

if (identical_columns) {
  print("All datasets have identical columns.")
} else {
  print("There are differences in column names across datasets.")
  # Print the differing columns for debugging
  lapply(all_columns, setdiff, colnames(df1))
}


# Rename the column in each dataframe
colnames(df1)[colnames(df1) == "X2015.PRICE.VAR...."] <- "PRICE_VAR"
colnames(df2)[colnames(df2) == "X2016.PRICE.VAR...."] <- "PRICE_VAR"
colnames(df3)[colnames(df3) == "X2017.PRICE.VAR...."] <- "PRICE_VAR"
colnames(df4)[colnames(df4) == "X2018.PRICE.VAR...."] <- "PRICE_VAR"
colnames(df5)[colnames(df5) == "X2019.PRICE.VAR...."] <- "PRICE_VAR"


# Check for proper renaming
grep("PRICE_VAR", colnames(df1), value = TRUE)

# Display the column names of the first dataset
colnames(df1)


# Get a statistical summary
summary(df1)

# Drop the first column from each dataframe
df1 <- df1[, -1]
df2 <- df2[, -1]
df3 <- df3[, -1]
df4 <- df4[, -1]
df5 <- df5[, -1]

# Check if the first column was removed
head(df1)

# Function to identify column types
get_column_types <- function(df) {
  sapply(df, function(col) {
    if (is.numeric(col)) {
      return("numeric")
    } else if (is.factor(col) || is.character(col)) {
      return("categorical")
    } else {
      return("other")
    }
  })
}

# Apply the function to your dataframe
column_types <- get_column_types(df1)

# Count the number of each type
type_counts <- table(column_types)

# Display the column types and their counts
print(type_counts)

# Convert the 'Sector' column to numeric using factor
df1$Sector <- as.numeric(as.factor(df1$Sector))
df2$Sector <- as.numeric(as.factor(df2$Sector))
df3$Sector <- as.numeric(as.factor(df3$Sector))
df4$Sector <- as.numeric(as.factor(df4$Sector))
df5$Sector <- as.numeric(as.factor(df5$Sector))

# Check the mapping of levels to numbers
levels <- levels(as.factor(df1$Sector))
print(levels)


# Adding the year column to each dataset, for future visualization purposes
df1$Year <- 2014
df2$Year <- 2015
df3$Year <- 2016
df4$Year <- 2017
df5$Year <- 2018


# Check for missing values in df1
sum(is.na(df1))  # Total number of missing values

# Check for missing values in each column
colSums(is.na(df1))


# Impute numeric and categorical columns for all dataframes
impute_missing <- function(df) {
  df <- data.frame(lapply(df, function(x) {
    if (is.numeric(x)) {
      x[is.na(x)] <- mean(x, na.rm = TRUE)  # Impute numeric with mean
    } else if (is.factor(x) || is.character(x)) {
      x[is.na(x)] <- get_mode(x[!is.na(x)])  # Impute categorical with mode
    }
    return(x)
  }))
  return(df)
}

df1 <- impute_missing(df1)
df2 <- impute_missing(df2)
df3 <- impute_missing(df3)
df4 <- impute_missing(df4)
df5 <- impute_missing(df5)

# Verify no missing values in df1
sum(is.na(df1))

# Verify no missing values across all dataframes
sapply(list(df1, df2, df3, df4, df5), function(df) sum(is.na(df)))

# Function to replace outliers with median
replace_outliers_with_median <- function(column) {
  Q1 <- quantile(column, 0.25, na.rm = TRUE)
  Q3 <- quantile(column, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  column[column < lower_bound | column > upper_bound] <- median(column, na.rm = TRUE)
  return(column)
}

# Apply to numerical columns for all dataframes
replace_outliers_in_dataframe <- function(df) {
  numerical_columns <- sapply(df, is.numeric)
  df[, numerical_columns] <- lapply(df[, numerical_columns], replace_outliers_with_median)
  return(df)
}

# Apply to each dataframe
df1_clean <- replace_outliers_in_dataframe(df1)
df2_clean <- replace_outliers_in_dataframe(df2)
df3_clean <- replace_outliers_in_dataframe(df3)
df4_clean <- replace_outliers_in_dataframe(df4)
df5_clean <- replace_outliers_in_dataframe(df5)

# Display the first 5 rows of the cleaned dataframe
head(df1_clean, 5)

# Check summary statistics for df1_clean
summary(df1_clean)

# Select specific numerical columns for boxplot
selected_columns <- c("Revenue", "EPS", "Net.Debt", "Market.Cap", "ROIC")


# Combine boxplots from multiple dataframes
par(mfrow = c(1, 2)) # Display two plots side by side
boxplot(df1_clean[, selected_columns], 
        main = "df1_clean", 
        col = "lightblue", 
        las = 2)

summary(df1_clean[, selected_columns])


# Concatenate all cleaned dataframes
data <- rbind(df1_clean, df2_clean, df3_clean, df4_clean, df5_clean)

# Display structure of the concatenated dataframe
str(data)

# Summary statistics for numerical columns only
summary(data[, sapply(data, is.numeric)])

# Number of rows and columns
nrow(data)
ncol(data)

# Check for missing values in the entire dataframe
sum(is.na(data)) 

# Check for missing values per column
colSums(is.na(data))



# Create a bar plot
ggplot(data, aes(x = as.factor(Year))) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Number of Records per Year", 
       x = "Year", 
       y = "Count of Records") +
  theme_minimal()

# Boxplot of PRICE_VAR
ggplot(data, aes(y = PRICE_VAR)) +
  geom_boxplot(fill = "lightgreen", color = "black", outlier.color = "red") +
  labs(title = "Boxplot of PRICE_VAR",
       y = "PRICE_VAR") +
  theme_minimal()

# Histogram of PRICE_VAR
ggplot(data, aes(x = PRICE_VAR)) +
  geom_histogram(bins = 50, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of PRICE_VAR",
       x = "PRICE_VAR",
       y = "Count") +
  theme_minimal()


# Feature engineering

# WITH PCA-----------------------
# Target variable
target <- data$PRICE_VAR
target

# Exclude non-numeric and target column from predictors
predictors <- data[, !(colnames(data) %in% c("PRICE_VAR", "Class"))]

constant_columns <- sapply(predictors, function(x) var(x, na.rm = TRUE) == 0)

predictors <- predictors[, !constant_columns]

# Ensure only numeric columns are used
predictors <- predictors[, sapply(predictors, is.numeric)]

# Standardize the data
predictors_scaled <- scale(predictors)

# Perform PCA
pca_result <- prcomp(predictors_scaled, center = TRUE, scale. = TRUE)

# Summary of PCA
summary_pca <- summary(pca_result)

# Display explained variance
variance_explained <- summary_pca$importance[2, ]
cumulative_variance <- cumsum(variance_explained)
print(cumulative_variance[1:120])  # Variance explained by the top 120 components


# Get PCA loadings
pca_loadings <- pca_result$rotation

# View loadings for the first few components
head(pca_loadings[, 1:5])  # Loadings for the first 5 components

# Find the top contributors to PC1
pc1_contributors <- sort(abs(pca_loadings[, 1]), decreasing = TRUE)
print(pc1_contributors)


# Modelling

# Extract the first 120 principal components
top_120_features <- as.data.frame(pca_result$x[, 1:120])

# Combine the PCA features with the target variable
data_pca <- cbind(top_120_features, PRICE_VAR = target)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
train_indices <- sample(1:nrow(data_pca), size = 0.8 * nrow(data_pca))
train_data <- data_pca[train_indices, ]
test_data <- data_pca[-train_indices, ]

# Separate predictors and target for train and test sets
x_train <- train_data[, 1:120]  
y_train <- train_data$PRICE_VAR

x_test <- test_data[, 1:120]
y_test <- test_data$PRICE_VAR


# Fit a LINEAR REGRESSION MODEL
model <- lm(PRICE_VAR ~ ., data = train_data)

# Summarize the model
summary(model)


# Make predictions on the testing set
predictions <- predict(model, newdata = test_data)

# Calculate evaluation metrics
mse <- mean((predictions - y_test)^2)  # Mean Squared Error
rmse <- sqrt(mse)  # Root Mean Squared Error
r2 <- 1 - (sum((predictions - y_test)^2) / sum((y_test - mean(y_test))^2))  # R-squared

# Print metrics
print(paste("MSE:", mse))
print(paste("RMSE:", rmse))
print(paste("R-squared:", r2))



# Linear regression with mi and multicollinearity applied

# Feature engineering with Mutual Information(MI)

data_regression <- data[, !(names(data) %in% c("Year", "Class"))]


# Select all numerical features excluding the target
features <- setdiff(names(data_regression), "PRICE_VAR")

# Compute Mutual Information (MI) for all features
mi_scores <- sapply(features, function(feature) {
  discretized_feature <- discretize(data_regression[[feature]], disc = "equalfreq", nbins = 10)  # Discretize feature
  mutinformation(discretized_feature, discretize(data_regression$PRICE_VAR, disc = "equalfreq", nbins = 10))  # Compute MI
})

# Convert MI results into a dataframe
mi_results <- data.frame(Feature = names(mi_scores), Mutual_Information = mi_scores)

# Sort features by MI score
mi_results <- mi_results %>% arrange(desc(Mutual_Information))

# Display the top 120 features
print(head(mi_results, 120))


selected_features <- intersect(mi_results$Feature[1:120], colnames(data_regression))

data_selected <- data_regression[, selected_features, drop=FALSE]


# Compute correlation matrix
cor_matrix <- cor(data_selected, use = "pairwise.complete.obs")

# Identify highly correlated pairs (absolute correlation > 0.8)
high_corr_pairs <- which(abs(cor_matrix) > 0.8 & lower.tri(cor_matrix), arr.ind = TRUE)

# Convert matrix index to feature names
correlated_vars <- data.frame(
  Var1 = rownames(cor_matrix)[high_corr_pairs[, 1]],
  Var2 = colnames(cor_matrix)[high_corr_pairs[, 2]],
  Correlation = cor_matrix[high_corr_pairs]
)

# Count how many times each variable appears in correlated pairs
correlation_counts <- as.data.frame(table(unlist(correlated_vars[, 1:2])))

# Identify variables with more than 3 correlations (to remove)
remove_vars <- correlation_counts$Var1[correlation_counts$Freq > 3]

# Remove these variables
filtered_features <- setdiff(selected_features, remove_vars)

# Handle pairs with only one correlation:
for (i in 1:nrow(correlated_vars)) {
  if (correlated_vars$Var1[i] %in% filtered_features & correlated_vars$Var2[i] %in% filtered_features) {
    mi1 <- mi_results$Mutual_Information[mi_results$Feature == correlated_vars$Var1[i]]
    mi2 <- mi_results$Mutual_Information[mi_results$Feature == correlated_vars$Var2[i]]
    
    # Remove the one with lower Mutual Information
    if (mi1 < mi2) {
      filtered_features <- setdiff(filtered_features, correlated_vars$Var1[i])
    } else {
      filtered_features <- setdiff(filtered_features, correlated_vars$Var2[i])
    }
  }
}

# Display the final list of selected features
print(length(filtered_features))  # Number of remaining features
print(filtered_features)  # Names of remaining features

# Linear Regression modelling with MI --------------------------------



# Ensure PRICE_VAR is the target variable
target <- "PRICE_VAR"

# Prepare dataset with selected 89 features
final_features <- c(filtered_features) 
data_final <- data_regression[, c(final_features, target), drop=FALSE]

# Split data into Training (80%) and Testing (20%)
set.seed(42)  # For reproducibility
trainIndex <- createDataPartition(data_final[[target]], p = 0.8, list = FALSE)
train_data <- data_final[trainIndex, ]
test_data <- data_final[-trainIndex, ]

# Build the Linear Regression Model
model <- lm(PRICE_VAR ~ ., data = train_data)

# Summary of the Model - Feature Importance and Significance
summary(model)

# Make Predictions
predictions <- predict(model, newdata = test_data)

# Model Evaluation Metrics
mse <- mean((predictions - test_data$PRICE_VAR)^2)  # Mean Squared Error
rmse <- sqrt(mse)  # Root Mean Squared Error
r2 <- 1 - (sum((predictions - test_data$PRICE_VAR)^2) / sum((test_data$PRICE_VAR - mean(test_data$PRICE_VAR))^2))  # R-squared

# Print Metrics
print(paste("MSE:", round(mse, 4)))
print(paste("RMSE:", round(rmse, 4)))
print(paste("R-squared:", round(r2, 4)))


# Visualize Top 20 Feature Importance based on p-values
coefficients_df <- as.data.frame(summary(model)$coefficients)
coefficients_df$Feature <- rownames(coefficients_df)
colnames(coefficients_df) <- c("Estimate", "Std_Error", "T_value", "P_value", "Feature")

# Select Top 20 Most Significant Features
coefficients_df <- coefficients_df[-1, ]
top_features <- coefficients_df[order(coefficients_df$P_value), ][1:20, ]

# Plot Feature Significance
ggplot(top_features, aes(x = reorder(Feature, P_value), y = -log10(P_value))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Feature Importance based on Significance (Linear Regression)",
       x = "Feature",
       y = "-log10(P-value)") +
  theme_minimal()





