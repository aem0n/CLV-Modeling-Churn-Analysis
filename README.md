# CLV-Modeling-Churn-Analysis
Predictive CLV and Churn pipeline using BG/NBD, Gamma-Gamma, and Random Forest. Features advanced cohort filtering and return-rate signal preservation to ensure statistical validity.
# Predictive Customer Analytics: RFM, CLV, and Churn Pipeline

This repository implements an end-to-end data science pipeline for e-commerce customer behavior analysis. The project moves from descriptive segmentation to predictive modeling, focusing on customer retention and future value estimation using the Online Retail II dataset.

## Core Analytical Modules

### 1. Customer Segmentation (RFM)
Utilizes Recency, Frequency, and Monetary metrics to categorize the customer base. 
* **Methodology:** K-Means clustering with optimal K selection via Silhouette and Elbow analysis.
* **Preprocessing:** Log-transformation and standard scaling to handle heavy-tailed distributions.



### 2. Predictive Customer Lifetime Value (CLV)
A probabilistic approach to determine the financial value of each customer over the next 6 months.
* **Models:** BetaGeoFitter (BG/NBD) for transaction frequency and GammaGammaFitter for monetary value.
* **Logic:** Differentiates between "age" (time since first purchase) and "recency" to predict future engagement.



### 3. Cohort Retention Analysis
A time-based analysis to track how different groups of customers (cohorts) stay active over several months.
* **Focus:** Visualizes the "stickiness" of the product through a mature cohort heatmap, filtering out incomplete recent data to avoid bias.

### 4. Leak-Free Churn Prediction
A machine learning approach to identify customers at risk of leaving.
* **Prevention of Data Leakage:** Features are engineered within an "observation window" and validated against a "performance window."
* **Algorithm:** Random Forest Classifier with optimized depth to prevent overfitting.
* **Key Features:** Incorporates return rates and average order frequency as primary churn signals.



## Business Action Matrix

The model outputs a strategic matrix to guide marketing interventions based on Churn Probability and Monetary Value:

| Segment | Risk | Value | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Top Priority** | High | High | Immediate retention campaigns and high-touch support. |
| **Loyal Customers** | Low | High | Inclusion in VIP programs and early access to new products. |
| **Low Value Churners** | High | Low | Automated re-engagement or passive monitoring. |
| **Loyal Low Value** | Low | Low | Cross-selling and up-selling to increase basket size. |

## Technical Implementation

### Prerequisites
The script requires Python 3.8+ and the following libraries:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn lifetimes openpyxl
