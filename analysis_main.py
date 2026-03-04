import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
try:
    from lifetimes.utils import customer_lifetime_value
except ImportError:
    from lifetimes.utils import _customer_lifetime_value as customer_lifetime_value
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "online_retail_II.xlsx")
sheet1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
sheet2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

merged_df = pd.concat([sheet1, sheet2], ignore_index=True)
print("Combined shape:", merged_df.shape)


# rows without Customer ID are useless for customer-level analysis, so dropping them
merged_df.dropna(subset=["Customer ID"], inplace=True)


is_return = merged_df["Invoice"].astype(str).str.startswith("C")
return_counts = merged_df[is_return].groupby("Customer ID")["Invoice"].nunique().rename("return_count")
total_invoices = merged_df.groupby("Customer ID")["Invoice"].nunique().rename("total_invoices")
return_rate_df = pd.concat([return_counts, total_invoices], axis=1).fillna(0)
return_rate_df["return_rate"] = return_rate_df["return_count"] / return_rate_df["total_invoices"]

# keep a snapshot of the raw (pre-filter) data so the churn feature block can compute
# return metrics for the observation window separately
raw_with_returns = merged_df.copy()

merged_df = merged_df[~is_return]

# some rows had negative or zero quantities/prices which don't make sense for purchase analysis
merged_df = merged_df[(merged_df["Quantity"] > 0) & (merged_df["Price"] > 0)]
print("Shape after cleaning:", merged_df.shape)


# outlier handling
def outlier_thresholds(dataframe, variable):
    low_limit = dataframe[variable].quantile(0.01)
    up_limit = dataframe[variable].quantile(0.99)
    return low_limit, up_limit

# instead of dropping outliers entirely I'm capping them at the thresholds so we don't lose data
# this way extreme values still contribute to the analysis but don't skew the results
def replace_with_thresholds(dataframe, variable):
    low, up = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low, variable] = low
    dataframe.loc[dataframe[variable] > up, variable] = up

replace_with_thresholds(merged_df, "Quantity")
replace_with_thresholds(merged_df, "Price")



merged_df["TotalPrice"] = merged_df["Quantity"] * merged_df["Price"]


# calculating RFM metrics
# adding 2 days to max date so that the most recent customer still gets a recency of at least 2
# (otherwise the most recent buyer would have recency=0 which can cause issues later)
today_date = merged_df["InvoiceDate"].max() + dt.timedelta(days=2)
print("Analysis Date:", today_date)

rfm_calculated = merged_df.groupby("Customer ID").agg(
    {"InvoiceDate": lambda x: (today_date - x.max()).days,   # recency
     "Invoice":     "nunique",                                # frequency
     "TotalPrice":  "sum"}                                    # monetary
)

rfm_calculated.columns = ["Recency", "Frequency", "Monetary"]
rfm_calculated = rfm_calculated.reset_index()

print("\nRFM first 5 rows:")
print(rfm_calculated.head())


# log transformation to handle skewness
rfm_calculated["Log_Recency"] = np.log1p(rfm_calculated["Recency"])
rfm_calculated["Log_Frequency"] = np.log1p(rfm_calculated["Frequency"])
rfm_calculated["Log_Monetary"] = np.log1p(rfm_calculated["Monetary"])

scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(rfm_calculated[["Log_Recency", "Log_Frequency", "Log_Monetary"]])
print("\nScaled features shape:", scaled_rfm.shape)


# finding the optimal number of clusters
k_values = range(2, 11)

wcss_list = []
sil_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
    model.fit(scaled_rfm)
    wcss_list.append(model.inertia_)
    sil_scores.append(silhouette_score(scaled_rfm, model.labels_))
    print(f"K={k:>2d}  |  WCSS={model.inertia_:>12.2f}  |  Silhouette={sil_scores[-1]:.4f}")

single_cluster = KMeans(n_clusters=1, init="k-means++", n_init=10, max_iter=300, random_state=42)
single_cluster.fit(scaled_rfm)
all_wcss = [single_cluster.inertia_] + wcss_list
all_k = range(1, 11)


# plotting elbow and silhouette side by side so I can compare them visually
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(all_k), all_wcss, marker="o", linewidth=2, color="#3498db")
axes[0].set_title("Elbow Method (WCSS)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("WCSS")
axes[0].set_xticks(list(all_k))
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_values), sil_scores, marker="s", linewidth=2, color="#e74c3c")
axes[1].set_title("Silhouette Scores", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(list(k_values))
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.show()


# I'm letting the silhouette decide instead of eyeballing the elbow, more objective this way
best_k = list(k_values)[np.argmax(sil_scores)]
print(f"\nOptimal K (best silhouette): {best_k}")

final_kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=10, max_iter=300, random_state=42)
final_kmeans.fit(scaled_rfm)
# adding the cluster labels back to the rfm dataframe so we know which group each customer is in
rfm_calculated["Cluster"] = final_kmeans.labels_



cluster_means = rfm_calculated.groupby("Cluster").agg(
    Recency_Mean=("Recency", "mean"),
    Frequency_Mean=("Frequency", "mean"),
    Monetary_Mean=("Monetary", "mean"),
    Count=("Customer ID", "count"),
).round(2)

print("\nCluster Summary:")
print(cluster_means)
print("\nRFM with Clusters (first 10 rows):")
print(rfm_calculated[["Customer ID", "Recency", "Frequency", "Monetary", "Cluster"]].head(10))



# this is different from my manual RFM above - here "frequency" means REPEAT purchases only
# (first purchase doesn't count), and T is the customer's "age" in days since first purchase
rfm_t_data = summary_data_from_transaction_data(
    merged_df,
    customer_id_col="Customer ID",
    datetime_col="InvoiceDate",
    monetary_value_col="TotalPrice",
    freq="D",
)

print("\nRFM-T first 5 rows:")
print(rfm_t_data.head())


# fitting BG/NBD model
# penalizer_coef=0.001 adds a small L2 penalty to stabilize the parameter estimates

bg_model = BetaGeoFitter(penalizer_coef=0.001)
bg_model.fit(rfm_t_data["frequency"], rfm_t_data["recency"], rfm_t_data["T"])

# predicting how many purchases each customer will make in the next 6 months (180 days)

rfm_t_data["expected_purchases_6m"] = bg_model.predict(
    t=180,
    frequency=rfm_t_data["frequency"],
    recency=rfm_t_data["recency"],
    T=rfm_t_data["T"],
)

print("\nBG/NBD - Top 10 by expected 6-month purchases:")
print(rfm_t_data.sort_values("expected_purchases_6m", ascending=False).head(10))


# gamma-gamma model for expected average profit per transaction
# important: this only works for customers with frequency > 0 because the model needs
# at least one repeat purchase to estimate spending behavior
# customers with frequency=0 only bought once, so we can't model their spending pattern
repeat_customers = rfm_t_data[rfm_t_data["frequency"] > 0].copy()

gg_model = GammaGammaFitter(penalizer_coef=0.001)
gg_model.fit(repeat_customers["frequency"], repeat_customers["monetary_value"])

# this predicts what each customer's average order value will be going forward
repeat_customers["expected_avg_profit"] = gg_model.conditional_expected_average_profit(
    repeat_customers["frequency"],
    repeat_customers["monetary_value"],
)



clv_result = customer_lifetime_value(
    bg_model,
    repeat_customers["frequency"],
    repeat_customers["recency"],
    repeat_customers["T"],
    repeat_customers["monetary_value"],
    time=6,           # 6 months
    freq="D",
    discount_rate=0.01
)

repeat_customers["CLV_6m"] = clv_result.values


# merging CLV back with the clustered rfm data
# using left join so we keep all customers, even ones without CLV (one-time buyers)
# those will just have NaN in the CLV column which is fine
rfm_with_clv = rfm_calculated.merge(
    repeat_customers[["CLV_6m", "expected_purchases_6m", "expected_avg_profit"]],
    left_on="Customer ID",
    right_index=True,
    how="left",
)

print("\nTop 10 Customers by CLV:")
print(
    rfm_with_clv.sort_values("CLV_6m", ascending=False)[
        ["Customer ID", "Recency", "Frequency", "Monetary", "Cluster", "CLV_6m"]
    ].head(10)
)


# average CLV per cluster - this is interesting to see which segment is most valuable
avg_clv_by_cluster = (
    rfm_with_clv.groupby("Cluster")["CLV_6m"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"CLV_6m": "Avg_CLV_6m"})
)

print("\nAverage 6-Month CLV per Cluster:")
print(avg_clv_by_cluster)


# monthly cohort analysis

merged_df["InvoiceMonth"] = merged_df["InvoiceDate"].dt.to_period("M")

# finding when each customer first purchased, this becomes their cohort
first_purchase_month = merged_df.groupby("Customer ID")["InvoiceMonth"].min().rename("CohortMonth")
merged_df = merged_df.merge(first_purchase_month, on="Customer ID")

# cohort index = how many months have passed since their first purchase

merged_df["CohortIndex"] = (
    (merged_df["InvoiceMonth"].dt.year - merged_df["CohortMonth"].dt.year) * 12
    + (merged_df["InvoiceMonth"].dt.month - merged_df["CohortMonth"].dt.month)
)


# building the retention table
# counting unique customers in each cohort x month-index combination
cohort_counts = (
    merged_df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
    .nunique()
    .reset_index(name="CustomerCount")
)

retention_pivot = cohort_counts.pivot(
    index="CohortMonth", columns="CohortIndex", values="CustomerCount"
)

# dividing by the first column (month 0 = original cohort size) to get retention %
# so column 0 is always 100%, and the rest shows what fraction came back
initial_cohort_size = retention_pivot.iloc[:, 0]
retention_pct = retention_pivot.divide(initial_cohort_size, axis=0) * 100

print("\nRetention Table (%) - first 5 cohorts:")
print(retention_pct.head().round(1))



# Recent cohorts haven't had enough calendar time to show later retention columns,
# producing misleading NaN / zero drop-offs.  By trimming them we ensure every
# remaining cohort has had at least 4 full months of observation.
last_cohort = retention_pct.index.max()
cutoff_cohort = last_cohort - 4  # PeriodIndex arithmetic: drops last 4 months
retention_pct_mature = retention_pct[retention_pct.index <= cutoff_cohort]

print(f"\nCohort maturity filter: keeping cohorts up to {cutoff_cohort} "
      f"({len(retention_pct_mature)} of {len(retention_pct)} cohorts)")

# retention heatmap (mature cohorts only)
plt.figure(figsize=(16, 10))
sns.heatmap(
    retention_pct_mature,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    linewidths=0.5,
    vmin=0,
    vmax=100,
    annot_kws={"size": 8},
    cbar_kws={"label": "Retention %"},
)
plt.title("Monthly Cohort Retention Analysis (%) – Mature Cohorts", fontsize=16, fontweight="bold")
plt.xlabel("Cohort Index (Months Since First Purchase)")
plt.ylabel("Cohort Month")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cohort_retention.png", dpi=150)
plt.show()


# interpreting the retention results (using mature cohorts only for fair comparison)
# comparing the first 3 cohorts (oldest) vs last 3 mature cohorts
# looking at months 1-3 specifically because that's the critical early period
print("\nCohort Retention Interpretation (mature cohorts):")
early_avg = retention_pct_mature.iloc[:3, 1:4].mean().mean()
late_avg = retention_pct_mature.iloc[-3:, 1:4].mean().mean()

print(f"Avg retention (months 1-3) for earliest 3 cohorts : {early_avg:.1f}%")
print(f"Avg retention (months 1-3) for latest  3 mature   : {late_avg:.1f}%")

if late_avg > early_avg:
    print("→ Newer cohorts show HIGHER short-term retention – retention is improving.")
elif late_avg < early_avg:
    print("→ Newer cohorts show LOWER short-term retention – retention is declining.")
else:
    print("→ Retention rates are roughly stable across cohorts.")


#churn
# the idea is to split the data by time so we simulate how it would work in real life.
# observation window: everything BEFORE the reference date (we build features from this)
# performance window: everything AFTER (we check if they came back or not)

max_invoice_date = merged_df["InvoiceDate"].max()
reference_date = max_invoice_date - pd.DateOffset(months=6)

observation_data = merged_df[merged_df["InvoiceDate"] <= reference_date].copy()
performance_data = merged_df[merged_df["InvoiceDate"] > reference_date].copy()

print(f"\nMax InvoiceDate      : {max_invoice_date}")
print(f"Reference Date       : {reference_date}")
print(f"Observation rows     : {len(observation_data):,}")
print(f"Performance rows     : {len(performance_data):,}")



# a customer is "churned" if they existed before the reference date but never came back
# only including customers whose FIRST purchase was before reference_date
# (we wouldn't know about customers who haven't started buying yet)
eligible_customers = observation_data.groupby("Customer ID")["InvoiceDate"].min()
eligible_customers = eligible_customers[eligible_customers <= reference_date].index

# checking which of those customers actually showed up in the performance window
active_after = performance_data[performance_data["Customer ID"].isin(eligible_customers)]["Customer ID"].unique()

# if they're NOT in the active list, they churned (Is_Churn = 1)
churn_labels = pd.DataFrame({"Customer ID": eligible_customers})
churn_labels["Is_Churn"] = (~churn_labels["Customer ID"].isin(active_after)).astype(int)
churn_labels = churn_labels.set_index("Customer ID")

print(f"\nCustomers in scope   : {len(churn_labels):,}")
print(f"Churn distribution:\n{churn_labels['Is_Churn'].value_counts()}")


# building features from ONLY the observation window (no data leakage)
# recency here is measured from the reference_date, not from today
# this is what we would actually know at the time of prediction
recency_feat = observation_data.groupby("Customer ID")["InvoiceDate"].max().apply(
    lambda x: (reference_date - x).days
).rename("Recency")

frequency_feat = observation_data.groupby("Customer ID")["Invoice"].nunique().rename("Frequency")

# recalculating TotalPrice for the observation window only
observation_data["TotalPrice_obs"] = observation_data["Quantity"] * observation_data["Price"]
monetary_feat = observation_data.groupby("Customer ID")["TotalPrice_obs"].sum().rename("Monetary")

# average days between orders helps capture purchasing rhythm
customer_timespan = observation_data.groupby("Customer ID")["InvoiceDate"].agg(
    lambda x: (x.max() - x.min()).days
)
avg_freq = (customer_timespan / frequency_feat.replace(0, np.nan)).rename("AvgOrderFreq")

num_unique_products = observation_data.groupby("Customer ID")["StockCode"].nunique().rename("UniqueCategories")

# return rate – computed from the raw (pre-filter) data so cancelled invoices are visible
# we restrict to the observation window to avoid data leakage
raw_obs = raw_with_returns[raw_with_returns["InvoiceDate"] <= reference_date].copy()
raw_obs_is_return = raw_obs["Invoice"].astype(str).str.startswith("C")
obs_return_counts = raw_obs[raw_obs_is_return].groupby("Customer ID")["Invoice"].nunique().rename("return_count")
obs_total_invoices = raw_obs.groupby("Customer ID")["Invoice"].nunique().rename("total_invoices")
obs_return_df = pd.concat([obs_return_counts, obs_total_invoices], axis=1).fillna(0)
customer_return_rate = (obs_return_df["return_count"] / obs_return_df["total_invoices"]).fillna(0).rename("ReturnRate")

# putting it all together into one feature matrix
# .loc[eligible_customers] ensures we only keep customers who were around before the cutoff
training_features = pd.concat(
    [recency_feat, frequency_feat, monetary_feat,
     avg_freq, num_unique_products, customer_return_rate,
     churn_labels["Is_Churn"]],
    axis=1,
).loc[eligible_customers].dropna()

print(f"\nLeak-free feature matrix: {training_features.shape}")


# training random forest
feature_cols = training_features.drop(columns="Is_Churn")
target_col = training_features["Is_Churn"]

# stratify=target_col ensures the train/test split has the same churn ratio as the full data
# without this, the test set might end up with way more/fewer churned customers by chance
training_data, testing_data, training_labels, testing_labels = train_test_split(
    feature_cols, target_col, test_size=0.25, random_state=42, stratify=target_col
)

# I set max_depth to 10 because anything higher was leading to overfitting on the training set
# (training accuracy was like 99% but test accuracy dropped)
# n_jobs=-1 uses all CPU cores to speed up training
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(training_data, training_labels)

predictions = rf_model.predict(testing_data)
print("\nClassification Report (Leak-Free):")
print(classification_report(testing_labels, predictions, target_names=["Active", "Churned"]))


# feature importance plot
feature_importances = pd.Series(rf_model.feature_importances_, index=feature_cols.columns).sort_values()

plt.figure(figsize=(10, 6))
feature_importances.plot(kind="barh", color=sns.color_palette("viridis", len(feature_importances)))
plt.title("Feature Importance – Leak-Free Churn Prediction", fontsize=14, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nTop predictors of churn (leak-free):")
for feat, imp in feature_importances[::-1].items():
    print(f"  {feat:<20s}  {imp:.4f}")


# Busineess Action Matrix
# predict_proba gives us the actual probability of churn (between 0 and 1) instead of just 0 or 1
# this is much more useful for business decisions because we can prioritize customers
# [:, 1] gets the probability of class 1 (churned)
churn_probabilities = rf_model.predict_proba(testing_data)[:, 1]

action_matrix_df = pd.DataFrame({
    "Customer ID": testing_data.index,
    "Churn_Probability": churn_probabilities,
    "Monetary": testing_data["Monetary"].values,
})

median_churn_prob = action_matrix_df["Churn_Probability"].median()
median_monetary = action_matrix_df["Monetary"].median()

# assigning each customer to one of 4 business segments:
# Top Priority Customer = high value + high risk → these are the customers we NEED to retain
# Loyal Customer = high value + low risk → our best customers, keep them happy
# Low Value Churner = low value + high risk → leaving but won't hurt us much financially
# Loyal Low Value Customer = low value + low risk → small but steady, no immediate action needed
def get_segment(row):
    high_risk = row["Churn_Probability"] >= median_churn_prob
    high_value = row["Monetary"] >= median_monetary
    if high_value and high_risk:
        return "Top Priority Customer"
    elif high_value and not high_risk:
        return "Loyal Customer"
    elif not high_value and high_risk:
        return "Low Value Churner"
    else:
        return "Loyal Low Value Customer"

action_matrix_df["Segment"] = action_matrix_df.apply(get_segment, axis=1)


# scatter plot showing the 4 quadrants
# I'm using a log scale for the Y-axis (Monetary) because the spending difference between
# customers is massive - some spend 50 and others spend 50,000. Without log scale the
# low value customers would all be squished at the bottom and you couldn't see anything.
# the dashed lines show the median thresholds that create the 4 quadrants
# red for danger (T.P.C), green for safe (L.C) 
segment_colors = {
    "Top Priority Customer":      "#e74c3c",
    "Loyal Customer":         "#2ecc71",
    "Low Value Churner": "#f39c12",
    "Loyal Low Value Customer":  "#3498db",
}

fig = px.scatter(
    action_matrix_df, 
    x="Churn_Probability", 
    y="Monetary",
    color="Segment",
    log_y=True,  # Log scale is a must because spending ranges from 50 to 50,000+
    color_discrete_map=segment_colors,
    hover_data=['Customer ID', 'Churn_Probability', 'Monetary'],
    title="Business Action Matrix (Interactive) - Churn Risk vs. Customer Value"
)
# adding the median lines to visually separate the four quadrants
fig.add_vline(x=median_churn_prob, line_dash="dash", line_color="grey")
fig.add_hline(y=median_monetary, line_dash="dash", line_color="grey")

# exporting to HTML so it can be shared with managers to open in any browser
fig.write_html("business_action_matrix.html")
print("Interactive matrix saved as 'business_action_matrix.html'. You can open this file in your browser.")

# summary table for each segment
# this is basically the executive summary 
segment_table = (
    action_matrix_df.groupby("Segment")
    .agg(
        Customer_Count=("Customer ID", "count"),
        Avg_Churn_Prob=("Churn_Probability", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Monetary=("Monetary", "sum"),
    )
    .round(2)
)

print("\nBusiness Action Matrix Summary:")
print(segment_table)

# this is the total money we stand to lose if the Top Priority Customers actually leave
# it's the strongest argument for investing in a retention campaign for this group
at_risk_revenue = action_matrix_df.loc[action_matrix_df["Segment"] == "Top Priority Customer", "Monetary"].sum()
print(f"\n⚠ Revenue at Risk (Top Priority segment): {at_risk_revenue:,.2f}")
