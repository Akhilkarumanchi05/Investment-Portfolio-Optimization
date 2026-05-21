# Investment Portfolio Optimization using Machine Learning

A machine-learning approach to building investment portfolios that **maximize returns while keeping risk at an acceptable level** — with assets auto-classified by risk and an interactive way for investors to tune their own risk appetite.

---

## Overview

Traditional portfolio methods (like Markowitz Mean-Variance Optimization) rely on simplifying assumptions that don't always hold in real markets. This project takes a data-driven alternative: it uses machine learning to predict asset returns and risk, classify assets into risk tiers, and optimize allocation across a diversified set of assets.

The result is a flexible tool that categorizes assets as **low-, medium-, or high-risk** and lets investors adjust risk percentages to fit their preferences.

## Dataset

- **Source:** [Kaggle](https://www.kaggle.com/) financial dataset
- **Span:** ~20 years of historical data across **55 assets**
- **Asset classes:** US stocks, bonds, gold, real estate, and currencies

This breadth captures a wide range of market dynamics, giving the models a robust foundation for training and testing.

## Approach

**Data preprocessing & EDA**
- Dropped the few missing values to preserve data integrity.
- Ran correlation analysis to identify and remove redundant, highly-correlated features (reducing multicollinearity).
- Visualized trends and volatility, with a focus on gold, stocks, and bonds.

**Risk & return engineering**
- Computed daily returns and **volatility** (252-day rolling standard deviation).
- Ran **Monte Carlo simulation (N=1,000)** with random portfolio weights to explore the risk/return space.
- Calculated key metrics: **Sharpe ratio**, cumulative returns, and annualized returns, and mapped the **efficient frontier**.

**Machine learning models**

| Technique | Role in the pipeline |
|---|---|
| Decision Trees / Random Forests | Asset selection — capturing complex, non-linear feature relationships |
| Support Vector Machines (SVM) | Fine-tuning portfolio allocation via optimal decision boundaries |
| Linear Regression | Predicting asset returns and risk (interpretable and transparent) |
| Classification models | Labeling assets as low- / medium- / high-risk |

## Key Results

- Built a portfolio optimizer that delivers strong **risk-adjusted returns** while letting investors control their risk level.
- Demonstrated the value of **diversification** in reducing overall portfolio risk.
- Produced interpretable outputs (regression coefficients, risk categories) that support clear, informed decision-making.

Visual outputs include returns-vs-volatility scatter plots, covariance heatmaps, Monte Carlo simulations, the efficient frontier, and actual-vs-predicted return plots.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Random Forest` · `SVM` · `Linear Regression` · `Monte Carlo Simulation` · `matplotlib` · `seaborn`

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Akhilkarumanchi05/Investment-Portfolio-Optimization.git
cd Investment-Portfolio-Optimization

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook
```

Run the cells in order, updating the dataset path to point to your local copy of the data.

## Future Work

- Add macro features (economic indicators, sentiment from financial news).
- Use more advanced risk measures beyond standard deviation (tail risk).
- Experiment with gradient boosting, time-series models, and reinforcement learning for **dynamic allocation** that adapts to changing markets.

---

*Built by [Akhil Karumanchi](https://github.com/Akhilkarumanchi05) · [LinkedIn](https://www.linkedin.com/in/akhil-karumanchi-56440922a/)*
