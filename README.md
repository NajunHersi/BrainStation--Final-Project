# Student Performance Factors — Predictive Modeling

A data science project exploring which factors most strongly influence student exam scores, using exploratory data analysis and linear regression modeling on the [Student Performance Factors dataset](https://www.kaggle.com/) by Muhammad Ahmad.

---

## Hypothesis

> *"Attendance, hours studied, and previous academic performance will be the strongest predictors of a student's final exam score, compared to environmental or socioeconomic variables."*

---

## Project Structure

```
├── run_model.ipynb                  # Main notebook: EDA + modeling pipeline
└── StudentPerformanceFactors.csv    # Source dataset (from Kaggle)
```

---

## Workflow Overview

### 1. Data Sourcing
- Dataset loaded via KaggleHub: `StudentPerformanceFactors.csv`
- Mix of academic, social, and environmental features alongside a continuous target variable (`Exam_Score`)

### 2. Data Cleaning
- Identified missing values in `Teacher_Quality`, `Parental_Education_Level`, and `Distance_From_Home`
- ~3.4% of rows contained at least one null — dropped for simplicity
- One-hot encoded all categorical columns using `pd.get_dummies(drop_first=True)`
- Converted all columns to integer dtype for model compatibility

### 3. Exploratory Data Analysis (EDA)
- **Distribution analysis** of exam scores (histogram + KDE)
- **Correlation heatmap** across all features
- **Regression plots** for the top three predictors vs. `Exam_Score`

Key correlations with `Exam_Score`:

| Feature | Correlation (r) |
|---|---|
| Hours Studied | 0.58 |
| Attendance | 0.45 |
| Previous Scores | 0.17 |

> Hours studied emerged as the strongest predictor — stronger than previous academic scores, which partially contradicted the initial hypothesis.

### 4. Modeling
- **Model:** Linear Regression (`sklearn`)
- **Split:** 80/20 train-test, `random_state=42`
- **Target:** `Exam_Score`
- **Features:** All remaining columns post-encoding

### 5. Evaluation

| Metric | Value |
|---|---|
| R² Score | ~0.73 |
| Mean Absolute Error | — |
| RMSE | — |
| Cross-Val R² (5-fold) | ~0.72 |

- Residual plot confirmed errors are randomly scattered around zero — no systematic pattern
- The model tends to underpredict high-scoring outliers, suggesting unmeasured factors (e.g., motivation, exam difficulty) influence top performers

### 6. Feature Importance
- Coefficients ranked by absolute value to identify the strongest drivers
- Hours studied, attendance, and certain encoded categorical features had the largest positive coefficients

---

## Key Findings

- **Hours studied** has the strongest positive influence on exam performance
- **Attendance** is the second most important factor
- **Previous scores**, while positively correlated, had a weaker effect than expected — current habits matter more than past performance
- The model explains ~73% of variance in exam scores, indicating solid predictive power with room to improve

---

## Next Steps

- Incorporate additional features (study methods, motivation, exam difficulty)
- Test non-linear models (Decision Trees, Random Forests) to capture interactions linear regression may miss
- Further feature engineering to improve predictive power

---

## Libraries Used

```
pandas, numpy, matplotlib, seaborn, scikit-learn
```

---

## Dataset

**Student Performance Factors** — by Muhammad Ahmad, available on [Kaggle](https://www.kaggle.com/).
