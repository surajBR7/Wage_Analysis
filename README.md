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
## **2. Visualize Data**

### **Histogram of Wage Distribution:**
Provides insights into wage frequencies.
```R
ggplot(data, aes(x = wage)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  ggtitle("Distribution of Wage") +
  xlab("Wage") +
  ylab("Frequency")
```
## **Boxplot of Wages by Education**

Examines wage disparities across education levels.

---

## **3. Feature Engineering**

Data transformations include:
- Scaling and normalizing variables for machine learning models.
- Creating dummy variables for categorical data.

---

## **4. Train Machine Learning Models**

Models implemented include:
- **Decision Trees:** Built using `rpart` and visualized with `rpart.plot`.
- **Lasso Regression:** Using the `glmnet` library for feature selection and model training.

```R
lasso_model <- glmnet(as.matrix(X_train), y_train, alpha = 1)
```
## **5. Evaluate Models**

### **Metrics for evaluation:**
- **Regression Metrics:** RMSE, R-squared.
- **Classification Metrics:** ROC-AUC, Accuracy.

### **Performance is visualized with:**
- Confusion matrices.
- ROC curves.

---

## **How to Run**

### **Install Required Libraries:**
Ensure all libraries are installed using the following:
```R
install.packages(c(
  "caret", "car", "readr", "dplyr", "ggplot2", "GGally",
  "gridExtra", "glmnet", "Metrics", "rpart", "rpart.plot",
  "pROC", "tidyr", "reshape2"
))
```

## **Run the Script**

### **Execute the finial.R script in an R environment (e.g., RStudio):**
```R
source("finial.R")
```

## **File Descriptions**

- **`finial.R`:** The main script containing all functions and workflow steps.
- **Dataset:** The dataset for wage prediction is not included. Ensure to use a CSV file compatible with the preprocessing pipeline.

---

## **Results**

### **Key Visual Insights:**
- Wage distribution and disparities by education level.
- Relationships between features and wage outcomes.

### **Model Performance:**
- **Regression RMSE:** Captures the error in wage prediction.
- **Classification ROC-AUC:** Evaluates the model’s ability to classify observations correctly.

---

## **Future Enhancements**

- Incorporate additional machine learning models such as Random Forests and XGBoost.
- Improve hyperparameter tuning for optimal performance.
- Extend the analysis to include more diverse datasets for generalization.

---

## **References**

- [caret Documentation](https://topepo.github.io/caret/index.html)
- [ggplot2 Documentation](https://ggplot2.tidyverse.org/)
- [glmnet Package](https://glmnet.stanford.edu/)
- [Rpart Package](https://cran.r-project.org/web/packages/rpart/index.html)
