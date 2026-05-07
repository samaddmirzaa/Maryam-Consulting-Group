# 📊 Maryam Consulting Group — Price Prediction & Regression Modeling

> A end-to-end regression modeling project built for Maryam Consulting Group (MCG), a price research firm. The goal is to accurately predict the price of goods across multiple industries while uncovering the key factors that drive pricing decisions.

---

## 🧭 Project Overview

As a Data Scientist at MCG, this project works through the full modeling lifecycle from raw data exploration to regularized regression and time series across datasets covering **computer prices**, **diamonds**, **health insurance**, **San Francisco apartment rents**, **power consumption**, and **airline passengers**.

| Phase | Focus |
|---|---|
| Explore | Understand distributions, correlations, and outliers |
| Prepare | Feature engineering, encoding, train/test splitting |
| Model | Simple → Multiple → Regularized regression |
| Evaluate | R², MAE across training and test sets |
| Select | Compare models and interpret the best fit |

---

## 📁 Repository Structure

```
├── notebooks/
│   ├── 01_EDA.ipynb                          # Exploratory Data Analysis
│   ├── 01_EDA_assignment.ipynb
│   ├── 02_simple_regression.ipynb            # Simple Linear Regression
│   ├── 02_simple_regression_assignment.ipynb
│   ├── 02_regression_case_health_insurance.ipynb
│   ├── 03_multiple_regression.ipynb          # Multiple Linear Regression
│   ├── 03_multiple_regression_assignment.ipynb
│   ├── 04_assumptions.ipynb                  # Regression Assumptions
│   ├── 04_assumptions_assignment.ipynb
│   ├── 05_validating_testing.ipynb           # Train/Validation/Test Splits
│   ├── 05_data_splitting_assignment.ipynb
│   ├── 06_feature_engineering_assignment.ipynb
│   ├── 07_regularized_regression.ipynb       # Ridge, Lasso, Elastic Net
│   ├── 07_regression_sanfrancisco_project.ipynb
│   ├── 08_regularization_assignment.ipynb
│   ├── 09_regularized_regression_sanfrancisco_project.ipynb
│   ├── 10_time_series_assignment.ipynb
│   └── 11_time_series_project.ipynb
├── Data/
│   ├── AirPassengers.csv                     # Monthly airline passenger counts (time series)
│   ├── Computers.csv                         # PC hardware specs & prices
│   ├── Diamonds_Prices2022.csv               # Diamond characteristics & prices
│   ├── insurance.csv                         # Health insurance charges
│   ├── madrid_weather.csv                    # Madrid historical weather data
│   ├── powerconsumption.csv                  # Power consumption (time series)
│   ├── sf_clean.csv                          # San Francisco apartment rentals
│   └── taco_stands.csv                       # Taco stand data
└── README.md
```

---

## 🧪 Methods & Models

### Baseline
- OLS Multiple Regression via `statsmodels`
- Log-transformed target (`log(price)`) for better linearity

### Feature Engineering
- Polynomial features: `hd²`, `hd³`
- One-hot encoding of categorical variables (`cd`, `multi`, `premium`)

### Regularization (with cross-validated alpha selection)
| Model | Train R² | Test R² | Notes |
|---|---|---|---|
| OLS Baseline | 0.805 | 0.811 | No regularization |
| Ridge (α=2.22) | 0.805 | **0.811** | Shrinks coefficients |
| Lasso (α=0.001) | 0.799 | 0.804 | Zeros out `const`, `hd3` |
| Elastic Net (α=0.001, L1=0.01) | 0.804 | **0.812** | Hybrid penalty |

### Key Findings
- `trend`, `hd`, and `ram` are the strongest price predictors
- Lasso eliminated `const` and `hd³`, confirming those features add little signal
- Ridge and Elastic Net achieved the best generalization on held-out data

---

## 🛠️ Tech Stack

- **Python 3.13**
- `pandas`, `numpy` — data manipulation
- `statsmodels` — OLS regression & summaries
- `scikit-learn` — train/test split, scaling, RidgeCV, LassoCV, ElasticNetCV
- `seaborn`, `matplotlib` — visualization

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/mcg-price-prediction.git
cd mcg-price-prediction
pip install -r requirements.txt
jupyter notebook
```

Open any notebook in the `notebooks/` folder and run cells sequentially.

---

## 📌 Objectives

1. **Explore** & visualize the data
2. **Prepare** the data for modelling
3. **Apply** regression algorithms
4. **Evaluate** model fit with R² and MAE
5. **Select** the best model and interpret it

---

*Built as part of the Maryam Consulting Group Data Science Department.*
