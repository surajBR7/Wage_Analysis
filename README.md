# Wage Prediction and Analysis using R

This project implements a data analysis and predictive modeling pipeline for wage prediction based on a dataset. The project utilizes various R libraries for data preprocessing, visualization, feature engineering, and machine learning.

---

## **Project Overview**

This project demonstrates the following steps:

1. **Data Loading and Preprocessing:**
   - Load the dataset from a CSV file.
   - Handle missing values, factorize categorical variables, and ensure clean data preparation.

2. **Data Visualization:**
   - Explore data distributions and relationships using visualizations such as histograms, boxplots, and scatter plots.

3. **Feature Engineering:**
   - Transform and scale data for modeling.
   - Encode categorical variables for regression and classification models.

4. **Modeling:**
   - Train machine learning models (e.g., Decision Trees, Lasso Regression) for wage prediction.
   - Evaluate model performance using metrics such as RMSE and ROC-AUC.

5. **Visualization of Results:**
   - Plot model diagnostics, feature importance, and performance metrics.

---

## **Technologies Used**

### **Libraries**
- **Data Manipulation:** `dplyr`, `tidyr`, `reshape2`
- **Visualization:** `ggplot2`, `GGally`, `gridExtra`
- **Modeling:** `caret`, `glmnet`, `rpart`
- **Metrics & Diagnostics:** `Metrics`, `pROC`

---

## **Project Workflow**

### **1. Load and Prepare the Data**
- The dataset is loaded using `read_csv()` and cleaned to ensure compatibility with modeling.
```R
wage_data <- read_csv(file_path) %>%
  mutate_if(is.character, as.factor) %>%
  select_if(~!is.factor(.) || length(levels(.)) > 1)
```
