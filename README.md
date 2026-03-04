# Customer Analytics: Segmentation, CLV, and Churn Prediction

This repository provides an end-to-end customer analytics pipeline. It processes transaction data to perform RFM segmentation, predicts future Customer Lifetime Value (CLV), analyzes retention cohorts, and uses machine learning to identify customers at risk of churn.

## Project Overview

The analysis follows a structured five-step workflow:

1.  **Data Preprocessing:** Cleaning of cancelled orders, handling missing values, and applying outlier capping (1st and 99th percentiles) to ensure statistical stability.
2.  **RFM Segmentation:** Using K-Means clustering to group customers by Recency, Frequency, and Monetary metrics.
3.  **CLV Forecasting:** Utilizing BG/NBD and Gamma-Gamma probabilistic models to estimate the expected number of transactions and average profit per customer over a 6-month horizon.
4.  **Cohort Analysis:** Generating monthly retention heatmaps with a maturity filter to evaluate long-term loyalty trends.
5.  **Predictive Churn Modeling:** A leak-free Random Forest classifier that calculates the churn probability for each customer using behavior-based features (Return Rate, Avg Order Frequency, etc.).

---



## Dataset

> [!IMPORTANT]  
> **Manual Data Setup Required:** Due to file size limitations, the dataset is **not** included in this repository. You must download and place it manually to run the analysis.

1. **Download:** Get the "Online Retail II" dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
2. **Rename:** Ensure the downloaded file is named exactly **`online_retail_II.xlsx`**.
3. **Placement:** Place the file in the root directory of this project (the same folder as the script).


---

## Technical Stack

* **Language:** Python 3.8+
* **Data Analysis:** `pandas`, `numpy`
* **Modeling:** `scikit-learn` (K-Means, Random Forest), `lifetimes` (BG/NBD, Gamma-Gamma)
* **Visualization:** `matplotlib`, `seaborn`, `plotly`

---

## Visual Outputs & Insights

### 1. K-Means Optimization
The optimal number of clusters is determined through a side-by-side comparison of the Elbow Method (WCSS) and Silhouette Scores.



### 2. Cohort Retention Heatmap
This matrix identifies when customer drop-offs occur. A maturity filter is applied to exclude incomplete data from recent months.



### 3. Churn Prediction Feature Importance
The Random Forest model ranks which behaviors (e.g., how recently they shopped or their return rate) are the strongest predictors of whether a customer will stop buying.



### 4. Business Action Matrix
An interactive scatter plot that segments customers into four quadrants: **Top Priority**, **Loyal**, **Low Value Churner**, and **Loyal Low Value**.
https://aem0n.github.io/CLV-Modeling-Churn-Analysis/business_action_matrix.html


---

## Setup and Usage

### Installation
Install the necessary dependencies using terminal:
```bash
pip install -r requirements.txt
Install the necessary dependencies using terminal:
```bash
pip install -r requirements.txt
