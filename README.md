# NYC-Green-Taxi-Trip-Duration-Prediction
## 🚀 Project Overview

This project focuses on building a machine learning model to accurately predict the **trip duration** for NYC Green Taxi trips. Accurate duration prediction is a critical component for taxi and ride-sharing services, aiding in more precise estimated time of arrival (ETA) calculations, dynamic pricing, and optimized resource allocation. This is a supervised learning task framed as a **regression problem**.

The entire machine learning lifecycle, from data cleaning and feature engineering to model training and hyperparameter optimization, was tracked using **MLflow**.

## ⚙️ Methodology

### 1. Data Source and Loading
The project utilized a dataset loaded from a CSV file named `trip.csv`.

### 2. Data Cleaning & Preprocessing
* **Target Variable Creation:** The target variable, **`trip_duration`** (in minutes), was engineered by calculating the difference between `lpep_dropoff_datetime` and `lpep_pickup_datetime`.
* **Missing Data Handling:** The `ehail_fee` column was dropped entirely as it contained 100% missing (null) values.
* **Data Type Conversion:** Date/time columns were converted to datetime objects, and key feature columns were converted to the memory-efficient `'category'` data type.

### 3. Feature Engineering
New features were extracted from the `lpep_pickup_datetime` to capture temporal patterns:
* `pickup_year`
* `pickup_month`
* `pickup_dayofweek`
* `pickup_day`
* `pickup_hour`
* `pickup_minute`
* `pickup_second`

### 4. Feature Selection
A robust feature selection approach was implemented using a voting system across multiple techniques to ensure a reliable and performant subset of features:

1.  **Recursive Feature Elimination (RFE):** Used with **Decision Tree**, **Random Forest**, and **Gradient Boosting** regressors.
2.  **Lasso with Cross-Validation (LassoCV):** Used to identify non-zero coefficient features.

The final set of features was determined by selecting those chosen by at least **three of the four** methods.

**The 10 Selected Features:**
* `RatecodeID`
* `PULocationID` (Pickup Location ID)
* `passenger_count`
* `trip_distance`
* `mta_tax`
* `improvement_surcharge`
* `pickup_month`
* `pickup_dayofweek`
* `pickup_day`
* `pickup_hour`

---

## 📈 Model Training and Evaluation

The core modeling phase involved two stages: initial baseline evaluation and subsequent hyperparameter tuning. All models are ensemble-based tree regressors, well-suited for high-dimensional, non-linear data.

### 1. Baseline Model Comparison

Five regression models were initially trained and evaluated using **R-squared ($R^2$)**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)**.

| Model | Train $R^2$ | Test $R^2$ | Test MAE (min) | Test MSE ($min^2$) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.975 | **0.858** | 2.274 | 33.931 |
| XGBoost Regressor | 0.976 | 0.850 | 2.468 | 35.753 |
| Gradient Boosting | 0.843 | 0.849 | 2.623 | 36.012 |
| Bagging Regressor | 0.964 | 0.840 | 2.462 | 38.191 |
| AdaBoost Regressor | 0.676 | 0.690 | 6.051 | 73.824 |

The **RandomForestRegressor** emerged as the strongest baseline model with a Test $R^2$ of **0.858**.

### 2. Hyperparameter Tuning

The model's Hyperparameters were tuned three times (Coarse and Fine tunning) using **Randomized Search Cross-Validation (RandomizedSearchCV) and Grid Search Cross-Validation (GridSearchCV)** on the top-performing models to optimize their performance.

| Model | Best Tuning Parameters | Best Test $R^2$ | Best Test MAE (min) | Best Test MSE ($min^2$) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | `{'subsample': 0.9, 'n_estimators': 150, 'max_depth': 10}` | **0.871** | **2.164** | **30.528** |
| Random Forest | `{'n_estimators': 150, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': 50}` | 0.863 | 2.228 | 32.565 |
| Gradient Boosting | `{'n_estimators': 50, 'min_samples_split': 10, 'max_features': 'log2', 'max_depth': None, 'learning_rate': 0.1}` | 0.841 | 2.752 | 37.866 |

## ✅ Conclusion and Best Model

The **Tuned XGBoost Regressor** is the final selected model for production, achieving the highest performance metrics after optimization:
* **Test $R^2$:** **0.871**
* **Test MAE:** **2.164 minutes**
* **Test MSE:** **30.528 $min^2$**

This result indicates that the model can explain **87.1%** of the variance in trip duration and has an average absolute prediction error of approximately **2.16 minutes**.

---

## 🛠️ Tools and Libraries

| Category | Tools/Libraries Used |
| :--- | :--- |
| **Language** | Python |
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib.pyplot`, `seaborn` |
| **Machine Learning** | `scikit-learn` (StandardScaler, OneHotEncoder, ColumnTransformer, RFE, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, LassoCV), `xgboost` |
| **MLOps/Tracking** | **MLflow** |

## 📞 Contact

Namshima B. Iordye

[Twitter](https://x.com/Namshima001?t=M2BjOSSyH8Q6IuAQz391qw&s=09)

[Email](iordyebarnabas12@gmail.com)

[Linkedin](https://www.linkedin.com/in/namshima-iordye-98ba51232)
