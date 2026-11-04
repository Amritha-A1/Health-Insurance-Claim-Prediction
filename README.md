# Health-Insurance-Claim-Prediction


## 📌 Project Overview

Health Insurance Claim Prediction is a Machine Learning project that aims to predict the medical insurance claim amount based on an individual's personal details, health conditions, and lifestyle factors.

Insurance claim amounts vary depending on factors such as age, BMI, smoking habits, medical conditions, and lifestyle. Predicting these amounts can help insurance companies understand potential claim costs and manage financial risks.

In this project, I performed data cleaning, Exploratory Data Analysis (EDA), data preprocessing, model building, hyperparameter tuning, and model evaluation. Multiple Machine Learning regression algorithms were implemented and compared to identify the best-performing model.

**Problem Type:** Supervised Machine Learning – Regression  
**Target Variable:** `claim`

---

## 📖 Project Storytelling – From Data to Prediction

### The Problem

Health insurance companies need to estimate the amount they may have to pay for an individual's medical claims. However, claim amounts can vary significantly from person to person.

The main question behind this project was:

**Can we predict an individual's health insurance claim amount using their personal information, medical conditions, and lifestyle factors?**

### Understanding the Data

To explore this problem, I worked with a health insurance dataset containing 15,000 records and 13 columns.

The dataset includes information about individuals, such as age, gender, weight, BMI, hereditary diseases, smoking status, diabetes, blood pressure, exercise habits, city, and occupation.

The `claim` column was selected as the target variable because it represents the insurance claim amount that needs to be predicted.

### The Journey

I started by exploring the dataset to understand its structure, identify missing values, and check for duplicate records.

During data cleaning, I identified missing values in the `age` and `bmi` columns, zero values in the `bloodpressure` column, and duplicate records. I handled these issues using appropriate preprocessing techniques.

Next, I performed Exploratory Data Analysis (EDA) to understand the dataset and examine relationships between the features and the target variable.

After preparing the data, I separated the independent and dependent variables, handled numerical and categorical features, and split the dataset into training and testing sets.

I then trained multiple Machine Learning regression models and evaluated their performance using different regression metrics.

Finally, I applied hyperparameter tuning to Random Forest and KNN using GridSearchCV to improve their performance.

### The Outcome

After comparing the evaluated models, the tuned Random Forest Regressor achieved the highest R² score and the lowest RMSE.

The tuned KNN model achieved the lowest MAE.

This project demonstrates how Machine Learning techniques can be applied to estimate health insurance claim amounts using personal and health-related information.

---

## 📊 Dataset Information

- **Dataset Name:** Health Insurance Dataset
- **Dataset File:** `healthinsurance.csv`
- **Original Rows:** 15,000
- **Original Columns:** 13
- **Target Variable:** `claim`
- **Problem Type:** Regression

### Dataset Features

| Column Name | Description |
|---|---|
| age | Age of the individual |
| sex | Gender of the individual |
| weight | Weight of the individual |
| bmi | Body Mass Index |
| hereditary_diseases | Hereditary disease information |
| no_of_dependents | Number of dependents |
| smoker | Smoking status |
| city | City of the individual |
| bloodpressure | Blood pressure level |
| diabetes | Diabetes information |
| regular_ex | Regular exercise information |
| job_title | Occupation of the individual |
| claim | Insurance claim amount (Target) |

---

## ⚙️ Project Workflow

### 1. Data Loading and Understanding

- Imported the dataset using Pandas.
- Examined the dataset structure and dimensions.
- Checked column names and data types.
- Used statistical summaries to understand numerical features.
- Identified missing values and duplicate records.

### 2. Data Cleaning

The following issues were identified:

- Missing values in the `age` column: 396
- Missing values in the `bmi` column: 956
- Zero values in the `bloodpressure` column: 703
- Duplicate records: 1,096

The following techniques were applied:

- Removed duplicate records.
- Treated zero values in `bloodpressure` as missing values.
- Used median imputation for missing numerical values.
- Used the most frequent value for missing categorical values.

**After removing duplicate records:**

- Rows: 13,904
- Columns: 13

### 3. Exploratory Data Analysis (EDA)

Performed Exploratory Data Analysis to understand the dataset and identify patterns.

- Analyzed numerical and categorical features.
- Examined the distribution of insurance claim amounts.
- Explored relationships between different features and the target variable.
- Used data visualizations to understand the dataset.

### 4. Feature Selection

Separated the dataset into independent and dependent variables.

- **X:** Independent variables (12 input features).
- **y:** Dependent variable (`claim`).

The `claim` column was selected as the target variable because the objective is to predict the insurance claim amount.

### 5. Data Preprocessing

Prepared the dataset for Machine Learning algorithms.

**Numerical Features:**

- Handled missing values using median imputation.
- Applied StandardScaler for feature scaling.

**Categorical Features:**

- Handled missing values using most frequent imputation.
- Applied OneHotEncoder to convert categorical values into numerical format.
- Used `handle_unknown="ignore"` to handle unseen categories.

After categorical encoding, the dataset contained **146 features**.

### 6. Train-Test Split

Split the dataset into training and testing sets.

- Training Size: 80%
- Testing Size: 20%
- Random State: 42

| Dataset | Number of Rows |
|---|---:|
| Training Data | 11,123 |
| Testing Data | 2,781 |

### 7. Model Building

Implemented and evaluated the following Machine Learning regression algorithms:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor (SVR)
6. K-Nearest Neighbors (KNN) Regressor
7. AdaBoost Regressor
8. XGBoost Regressor

### 8. Model Evaluation

Evaluated model performance using the following regression metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error):** Measures the average squared difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Measures prediction error in the same unit as the target variable.
- **R² Score:** Measures how well the model explains the variation in the target variable.

### 9. Hyperparameter Tuning

Applied GridSearchCV to improve the performance of:

- Random Forest Regressor
- K-Nearest Neighbors (KNN) Regressor

Used 5-fold cross-validation and negative RMSE as the scoring method to identify suitable hyperparameters.

---

## 📈 Final Model Results

The following table shows the performance comparison of all evaluated models.

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Random Forest (Tuned) | 556.13 | 4,196,713.81 | 2,048.59 | 0.97196 |
| KNN (Tuned) | 423.75 | 4,693,933.81 | 2,166.55 | 0.96863 |
| Decision Tree | 463.60 | 6,769,718.00 | 2,601.87 | 0.95476 |
| XGBoost | 1,623.60 | 8,600,155.00 | 2,932.60 | 0.94253 |
| Gradient Boosting | 2,758.88 | 20,546,680.00 | 4,532.84 | 0.86270 |
| Linear Regression | 3,930.87 | 35,636,070.00 | 5,969.60 | 0.76186 |
| AdaBoost | 8,798.76 | 88,630,270.00 | 9,414.36 | 0.40773 |
| SVR | 8,235.77 | 161,814,100.00 | 12,720.62 | -0.08131 |

---

## 💡 Key Insights from the Project

The model evaluation provided several useful insights:

1. **Best Overall Model:** The tuned Random Forest Regressor achieved the highest R² score of 0.97196 and the lowest RMSE of 2,048.59 among the evaluated models.

2. **Lowest MAE:** The tuned KNN model achieved the lowest MAE of 423.75, indicating the lowest average absolute prediction error.

3. **Model Comparison:** Different regression algorithms produced different prediction results. This highlights the importance of testing and comparing multiple models.

4. **Hyperparameter Tuning:** GridSearchCV helped identify suitable hyperparameters for Random Forest and KNN through 5-fold cross-validation.

5. **SVR Performance:** SVR achieved a negative R² score of -0.08131, indicating that it performed worse than a simple mean-based prediction baseline on the test data.

6. **Data Preprocessing:** Handling missing values, duplicate records, and categorical features was an important part of preparing the dataset for model training.

### What I Learned

Through this project, I gained practical experience in:

- Understanding and cleaning real-world datasets.
- Handling missing values and duplicate records.
- Performing Exploratory Data Analysis.
- Preparing numerical and categorical features.
- Building regression models using Scikit-learn.
- Comparing models using evaluation metrics.
- Applying GridSearchCV for hyperparameter tuning.
- Interpreting model performance and drawing conclusions.

---

## 🎯 Final Conclusion

This project demonstrates how Machine Learning can be used to predict health insurance claim amounts based on personal, medical, and lifestyle information.

After evaluating eight regression algorithms, the tuned Random Forest Regressor performed best overall based on its R² score and RMSE. The tuned KNN model achieved the lowest MAE.

The project helped me understand the complete Machine Learning workflow, from data exploration and preprocessing to model training, hyperparameter tuning, and evaluation.

It also provided practical experience in comparing different algorithms and selecting a suitable model based on evaluation metrics.

---

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Boosting Algorithm:** XGBoost
- **Development Environment:** Jupyter Notebook
- **Web Application:** Streamlit (if included in the repository)

---

## Live Demo

The Health Insurance Claim Prediction application is deployed using Streamlit Cloud.

**Live Application:** [Click here to access the Streamlit App](https://amritha-a1-health-insurance-claim-prediction-app-ru7mfq.streamlit.app/)

You can use the live application to enter personal and medical details and predict the health insurance claim amount.

## How to Run the Project Locally

**1. Clone the Repository**
```bash
git clone https://github.com/Amritha-A1/Health-Insurance-Claim-Prediction.git
```

**2. Navigate to the Project Folder**
```bash
cd Health-Insurance-Claim-Prediction
```

**3. Install Required Libraries**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit Application**
```bash
streamlit run app.py
```

The application will open in your web browser at:
`http://localhost:8501`



## 📁 Project Structure

```text
Health-Insurance-Claim-Prediction/
│
├── Health_Insurance.ipynb
├── healthinsurance.csv
├── app.py
├── requirements.txt
├── random_forest_model.pkl
├── preprocessor.pkl
└── README.md
