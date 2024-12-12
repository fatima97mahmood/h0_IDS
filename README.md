# ML-Driven Predictive Analysis for Loans

- **Fatima Mahmood**
- **Danni Zhang**
- **Romet Vislapuu**

## Project Overview
This project applies machine learning (ML) techniques to predict loan approval outcomes based on demographic, financial, and credit-related features. The goal is to classify whether a loan application is approved or denied by analyzing synthetic data inspired by real-world credit and financial datasets.

The project aims to:
- Predict binary loan approval outcomes.
- Identify key predictors influencing loan decisions.
- Apply various machine learning algorithms to achieve high predictive accuracy.

## Motivation
Loan approval is a vital process in financial services, traditionally dependent on manual evaluation. This project utilizes machine learning to predict loan approval status, automate decisions, enhance accuracy, and reduce bias. It streamlines operations, cuts costs, and delivers quicker, fairer outcomes for lenders and applicants.

## Dataset
This dataset is a synthetic version ([Source](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)) inspired by the original Credit Risk dataset ([Source](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)) on Kaggle, enriched with additional variables derived from Financial Risk for Loan Approval data ([Source](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval)). It serves as a valuable resource for predicting the loan status of potential applicants (approved or not approved), encompassing demographic, financial, and credit-related features. With 45,000 observations and 13 features, all complete and without missing values, the dataset provides a solid foundation for detailed analysis and modeling.

The dataset includes both categorical and numerical features such as:
- **Demographic Features**: Age, gender, education level, home ownership.
- **Financial Features**: Annual income, years of employment, loan amount requested, interest rate.
- **Credit Features**: Credit score, previous loan defaults, length of credit history.

The goal is to predict the likelihood of loan approval based on these attributes.

## Modelling
Our goal was to build prediction models for a binary classification task, where we discovered that only about 22.23% of applicants' loans were approved, indicating an imbalanced dataset.

### Dealing with Imbalanced Data
We applied **SMOTE** and used class weights during model training to address this.

### Model Selection and Training
We tested multiple machine learning models, performed hyperparameter tuning, and utilized cross-validation to ensure the models' robustness, aiming for improved performance despite the imbalanced data distribution.

## Machine Learning Methods
We employed several machine learning techniques to build predictive models. These methods are as follows:

1. **Decision Tree**:
   - Used for classification tasks to create a model that predicts loan approval.
   - Decision trees are interpretable, helping us understand the decision-making process.

2. **Random Forest**:
   - An ensemble method that builds multiple decision trees and combines their predictions for improved accuracy.
   - Reduces overfitting and increases model robustness.

3. **Logistic Regression**:
   - A statistical method that provides a probability estimate for loan approval.
   - Used for its simplicity and efficiency in binary classification.

4. **Support Vector Machine (SVM)**:
   - A classifier that finds the optimal hyperplane to separate loan approval outcomes.
   - Effective for high-dimensional data and works well with non-linear decision boundaries.

5. **Neural Networks**:
   - A deep learning approach to model complex relationships in the data.
   - Useful for capturing non-linear patterns and interactions between features.

6. **XGBoost**:
   - A gradient boosting algorithm popular for classification tasks due to its speed and performance.
   - Focuses on building strong predictive models by combining weak models in an iterative manner.

## Key Considerations
Our data was collected from Kaggle’s Loan Approval Classification dataset, consisting of 45,000 observations and 13 features. Fortunately, the dataset contains no missing values.

### Removing Outliers
We began by removing outliers, such as loan applicants with ages exceeding 100 years. Next, we addressed skewed data by applying a log transformation to variables like “annual income” and used the standard deviation method to identify and remove remaining outliers.

### Feature Engineering
Finally, we converted categorical variables like “Education” and “Loan_intent” into numerical values or dummy variables.

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/fatima97mahmood/loan-approval-prediction.git
2. Install dependencies: You can install the necessary dependencies using pip:
   ```bash
   pip install -r requirements.txt
3. Run the Jupyter Notebook: Open the Loan_Approval_Prediction.ipynb Jupyter notebook to start the analysis:
   ```bash
   jupyter notebook Loan_Approval_Prediction.ipynb

## Model Persistence with Pickle

This project uses **Pickle files** to save and load pretrained models, providing two options:

1. **Import Pretrained Models**: Quickly use the models for predictions without retraining, saving time and computational resources.
2. **Rerun Models in the Notebook**: Option to retrain the models from scratch for fine-tuning or experimentation, though this requires more computational effort.

The Pickle files offer a balance between **efficiency** (by loading pretrained models) and **flexibility** (by allowing retraining as needed).

## Files and Folders

- **Loan_Approval_Prediction.ipynb**: The main Jupyter notebook where the data analysis, preprocessing, and model training are done.
- **data/loan_data.csv**: The dataset used for training and testing the models.
- **models/**: Directory containing `model.pkl` files that have the saved version of the trained model (e.g., RandomForest, DecisionTree, etc.) and Keras Tuner files where the results of the hyperparameter search are stored, including the best configurations for the neural network model.
      ***NB!*** Due to GitHubs file size limitations, the models are available through this [link](https://drive.google.com/drive/folders/1ea5G8U2NmIiKeSg1Z5uhbesHNkp2_sWB?usp=sharing).
- **figures/**: Folder for storing any generated plots or visualizations (e.g., ROC curve, feature importance).

## Evaluation Metrics

We evaluate the performance of the models based on the following metrics:

- **Accuracy**: Proportion of correct predictions.
- **ROC AUC**: Area under the Receiver Operating Characteristic curve, which evaluates the model's ability to distinguish between classes.
- **Precision**: Measures the proportion of positive predictions that are actually correct.
- **Recall**: Measures the proportion of actual positives that were correctly identified.
- **F1-Score**: Harmonic mean of Precision and Recall, giving a balanced evaluation of both metrics.

## Contributions

- **Fatima Mahmood**: Focused on implementing and tuning Neural Networks and XGBoost models.
- **Danni Zhang**: Worked on Logistic Regression and SVM, and model evaluation.
- **Romet Vislapuu**: Implemented Decision Tree and Random Forest.

All members worked on business and data understanding, visualizations, and feature engineering.
