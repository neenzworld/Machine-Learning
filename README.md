# Machine Learning for Predictive Social Science

**Comparing OLS, LASSO Regularization, and Random Forests in Health Trajectories**

Author: **Yaw Dgyal**

---

## Overview

This workbook compares OLS, LASSO, and Random Forest models on global life expectancy trajectories using the `tidymodels` framework, walking through cross validation, residual diagnostics, and permutation based variable importance to demonstrate when each modeling strategy earns its keep.

---

## Prerequisites

Before attempting this workbook, you should already know how to:

- Perform data wrangling with **`dplyr`** and **`tidyr`**
- Build layered visualizations with **`ggplot2`**
- Fit and interpret basic **OLS regressions**
- Understand the concepts of **training and test splits** and **overfitting**

---

## Learning Objectives

By completing this workbook, you will develop competence in:

1. Building reproducible machine learning pipelines with the **`tidymodels`** framework
2. Defining **preprocessing recipes** that prevent data leakage across train, cross validation, and test sets
3. Fitting and comparing **OLS**, **LASSO**, and **Random Forest** models on the same data
4. Evaluating models with **RMSE**, **MAE**, and **R squared** across stratified cross validation folds
5. Interpreting **predicted versus actual plots**, **residual histograms**, **LASSO trace paths**, and **permutation variable importance**
6. Choosing between explanatory and predictive modeling strategies based on your research goal

---

## Repository Structure

```
.
├── README.md
├── machine-learning.qmd              # Main Quarto workbook
├── data/
│   └── life_expectancy_data.csv      # OWID cross-sectional life expectancy panel
└── reports/                          # Rendered HTML output (optional)
    └── machine-learning.html
```

---

## Data Source

| Source | Description | Link |
|---|---|---|
| **Our World in Data (OWID)** | National life expectancy panel across countries and years | https://ourworldindata.org/life-expectancy |

The workbook engineers three lag features from the raw panel, namely life expectancy one year ago, five years ago, and the five year change, so that past values serve as predictors of current values.

---

## Required R Packages

```r
install.packages(c(
  "tidyverse",
  "tidymodels",
  "ranger",
  "glmnet",
  "scales"
))
```

> **Note:** The `vip` package was archived from CRAN on 2026-07-13. It is now hosted on the maintainer r universe repository. The setup chunk installs it automatically with:
>
> ```r
> install.packages(
>   "vip",
>   repos = c("https://bgreenwell.r-universe.dev", "https://cloud.r-project.org")
> )
> ```

---

## Workflow Summary

| Section | Purpose |
|---|---|
| **Section 1** | Theoretical motivation contrasting explanatory inference with predictive modeling |
| **Section 2** | Dynamic path resolution, OWID ingestion, and lag feature engineering |
| **Section 3** | Stratified 80 20 train test split and five fold cross validation architecture |
| **Section 4** | Preprocessing recipes and model specifications for OLS, LASSO, and Random Forest |
| **Section 5** | Cross validated model training and metric collection |
| **Section 6** | Five diagnostic visualizations with interpretation guides |
| **Section 7** | Decision tree flowchart for choosing between modeling strategies |
| **Section 8** | Canonical references for further reading |

---

## How to Render

From the project root:

```bash
quarto render machine-learning.qmd
```

Or open the `.qmd` file in **RStudio** and click **Render**.

The workbook expects `life_expectancy_data.csv` to be located in either `data/` or `../data/` relative to the working directory. A home directory fallback is included for local development but should be removed or generalized before distributing the workbook to others.

---

## Key Concepts

- **Explanatory inference** versus **predictive performance**: two different scientific goals that call for different tools
- **Cross validation**: five fold stratified resampling to estimate out of sample error without touching the test set
- **LASSO regularization**: L1 penalty that shrinks uninformative coefficients to zero and performs automatic feature selection
- **Random Forest**: non parametric ensemble of decision trees that captures non linear interactions
- **Permutation importance**: how much a model relies on each predictor, measured by shuffling values and observing the damage to accuracy
- **Data leakage**: the silent error of preprocessing on the full dataset before splitting, which inflates test set performance

---

## Common Pitfalls Covered in the Workbook

Each section includes a Common Pitfalls callout box explaining the traps that most applied machine learning projects fall into, including test set leakage, forgotten seeds, unstratified splits, positional lagging errors, hardcoded hyperparameters, and single metric reporting.

---

## License

For educational use. Please cite Our World in Data as the original data provider.
